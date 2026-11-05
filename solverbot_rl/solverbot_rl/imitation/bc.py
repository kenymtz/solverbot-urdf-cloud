"""Behavior Cloning sobre el ACTOR de un modelo Stable-Baselines3 SAC (misma arquitectura que usará el fine-tuning).

No se entrena una red aparte ni se copian pesos entre arquitecturas: se construye un `SAC` real con la política
`BCSACPolicy` y se ajustan por aprendizaje supervisado los parámetros de `model.actor` (latent_pi, mu, log_std).
El crítico queda sin entrenar (lo inicializa/precalienta el fine-tuning).

Actor SAC de SB3: a = tanh(u), u ~ N(mu(s), exp(log_std(s))). La acción determinista es tanh(mu(s)).
Pérdida:
  * media:  MSE( tanh(mu(s)), a_experto )                      en el espacio de acciones normalizadas [-1, 1]
  * dispersión: NLL gaussiana de atanh(clip(a_experto, ±0.995)) con mu DETENIDA -> log_std aprende el error residual
    en el espacio previo a tanh. Junto con el término constante -sum log(1 - a^2), es la verosimilitud exacta de la
    gaussiana comprimida por tanh que usa SAC, de modo que la política estocástica inicial del fine-tuning es
    coherente con el BC (no un log_std arbitrario).

Normalización de la observación: `NormalizedObs` (features extractor) con media/desvío FIJOS del conjunto de
entrenamiento guardados como buffers dentro del modelo (se guardan en el .zip; el modelo recibe la observación cruda
del entorno). Las componentes ya acotadas en [-1, 1] o binarias (pinza, detección válida, acción previa, meta) no se
reescalan. `actor_mask` permite ocultar componentes AL ACTOR (por defecto la acción previa: evita la "política
copiona" de la clonación de comportamiento, que aprende a repetir a_{t-1}); el crítico siempre ve la observación
completa (la penalización por variación de acción depende de a_{t-1}).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from .rollout import ACT_DIM, OBS_DIM, OBS_SLICES

try:
    import torch
    import torch.nn as nn
    from stable_baselines3 import SAC
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.sac.policies import SACPolicy
except ImportError as exc:  # pragma: no cover
    raise ImportError(f'solverbot_rl.imitation.bc requiere torch y stable-baselines3: {exc}')

UNSCALED = ('gripper', 'obj_valid', 'prev_action', 'goal')       # ya acotadas / binarias: media 0, desvío 1
LABEL_CLIP = 0.995


def _indices(names) -> list[int]:
    out = []
    for n in names:
        s = OBS_SLICES[n]
        out.extend(range(s.start, s.stop))
    return out


def compute_normalization(obs: np.ndarray, std_floor: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    mean = obs.mean(axis=0).astype(np.float64)
    std = np.maximum(obs.std(axis=0), std_floor).astype(np.float64)
    keep = _indices(UNSCALED)
    mean[keep], std[keep] = 0.0, 1.0
    return mean, std


def default_actor_mask(hide=('prev_action',)) -> list[float]:
    m = np.ones(OBS_DIM)
    m[_indices(hide)] = 0.0
    return m.tolist()


class NormalizedObs(BaseFeaturesExtractor):
    """(obs - mean) / std, recortado a ±10, multiplicado por una máscara fija."""

    def __init__(self, observation_space: spaces.Box, mean, std, mask=None):
        super().__init__(observation_space, features_dim=int(np.prod(observation_space.shape)))
        n = self.features_dim
        self.register_buffer('mean', torch.as_tensor(np.asarray(mean, np.float32).reshape(n)))
        self.register_buffer('std', torch.as_tensor(np.asarray(std, np.float32).reshape(n)))
        self.register_buffer('mask', torch.as_tensor(np.ones(n, np.float32) if mask is None
                                                     else np.asarray(mask, np.float32).reshape(n)))

    def forward(self, obs):
        return torch.clamp((obs.flatten(1) - self.mean) / self.std, -10.0, 10.0) * self.mask


class BCSACPolicy(SACPolicy):
    """SACPolicy cuyo crítico usa la misma normalización pero SIN la máscara del actor."""

    def make_critic(self, features_extractor=None):
        if features_extractor is None:
            kw = dict(self.features_extractor_kwargs)
            kw['mask'] = None
            features_extractor = self.features_extractor_class(self.observation_space, **kw).to(self.device)
        return super().make_critic(features_extractor)


class SpacesEnv(gym.Env):
    """Entorno mínimo con los espacios de SolverBotPickPlaceEnv (31 obs, 7 acciones) para construir/cargar modelos
    sin simulador. No se usa para recolectar experiencia."""

    def __init__(self):
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(OBS_DIM, np.float32), {}

    def step(self, action):
        return np.zeros(OBS_DIM, np.float32), 0.0, True, False, {}


def make_sac(env, mean, std, actor_mask=None, seed: int = 0, net_arch=(256, 256), **sac_kwargs) -> 'SAC':
    pk = dict(net_arch=list(net_arch), features_extractor_class=NormalizedObs,
              features_extractor_kwargs={'mean': np.asarray(mean).tolist(), 'std': np.asarray(std).tolist(),
                                         'mask': None if actor_mask is None else list(actor_mask)})
    kw = dict(learning_rate=3e-4, buffer_size=200_000, batch_size=256, learning_starts=0, train_freq=1,
              gradient_steps=1, verbose=0, device='cpu')
    kw.update(sac_kwargs)
    return SAC(BCSACPolicy, env if env is not None else SpacesEnv(), policy_kwargs=pk, seed=seed, **kw)


def actor_outputs(actor, obs_t):
    mean_actions, log_std, _ = actor.get_action_dist_params(obs_t)
    return mean_actions, log_std


def bc_losses(actor, obs_t, act_t, w_nll: float = 0.1):
    mu, log_std = actor_outputs(actor, obs_t)
    mse = ((torch.tanh(mu) - act_t) ** 2).mean()
    u = torch.atanh(torch.clamp(act_t, -LABEL_CLIP, LABEL_CLIP))
    nll = (0.5 * ((u - mu.detach()) / log_std.exp()) ** 2 + log_std).mean()
    return mse + w_nll * nll, mse, nll


@torch.no_grad()
def evaluate_actor(actor, obs: np.ndarray, act: np.ndarray) -> dict:
    actor.eval()
    o = torch.as_tensor(obs, dtype=torch.float32)
    a = torch.as_tensor(act, dtype=torch.float32)
    mu, log_std = actor_outputs(actor, o)
    pred = torch.tanh(mu)
    err = (pred - a).numpy()
    actor.train()
    return {'mse': float((err ** 2).mean()), 'mae': float(np.abs(err).mean()),
            'mse_per_dim': (err ** 2).mean(axis=0).round(5).tolist(),
            'mse_arm_unsat': float((err[:, :6][np.abs(act[:, :6]) < 0.99] ** 2).mean())
            if (np.abs(act[:, :6]) < 0.99).any() else None,
            # concordancia de signo solo donde el experto se mueve de verdad (|a| > 0.1); con a = 0 no hay signo
            'sign_agreement_arm_moving': float(np.mean(np.sign(pred[:, :6].numpy())[np.abs(act[:, :6]) > 0.1]
                                                       == np.sign(act[:, :6])[np.abs(act[:, :6]) > 0.1])),
            'mean_std': float(log_std.exp().mean())}


def train_bc(model, obs: np.ndarray, act: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, *,
             epochs: int = 400, batch_size: int = 256, lr: float = 1e-3, weight_decay: float = 1e-5,
             w_nll: float = 0.1, patience: int = 60, seed: int = 0, log_path: str | None = None,
             sample_weights: np.ndarray | None = None) -> dict:
    """Ajusta model.actor. Early stopping por MSE de validación (episodios separados); restaura el mejor estado."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    actor = model.actor
    opt = torch.optim.Adam(actor.parameters(), lr=lr, weight_decay=weight_decay)
    o_tr = torch.as_tensor(obs[train_idx], dtype=torch.float32)
    a_tr = torch.as_tensor(act[train_idx], dtype=torch.float32)
    p = None
    if sample_weights is not None:
        p = sample_weights[train_idx] / sample_weights[train_idx].sum()
    best, best_state, best_epoch, hist = np.inf, None, -1, []
    n = len(train_idx)
    t0 = time.time()
    for ep in range(epochs):
        idx = rng.choice(n, n, p=p) if p is not None else rng.permutation(n)
        tot = []
        for k in range(0, n, batch_size):
            b = idx[k:k + batch_size]
            loss, mse, nll = bc_losses(actor, o_tr[b], a_tr[b], w_nll)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot.append([loss.item(), mse.item(), nll.item()])
        tr = np.mean(tot, axis=0)
        va = evaluate_actor(actor, obs[val_idx], act[val_idx]) if len(val_idx) else {'mse': tr[1]}
        rec = {'epoch': ep, 'train_loss': float(tr[0]), 'train_mse': float(tr[1]), 'train_nll': float(tr[2]),
               'val_mse': va['mse'], 'val_mae': va.get('mae'), 'val_mean_std': va.get('mean_std')}
        hist.append(rec)
        if log_path:
            with open(log_path, 'a') as fh:
                fh.write(json.dumps(rec) + '\n')
        if va['mse'] < best - 1e-6:
            best, best_epoch = va['mse'], ep
            best_state = {k: v.detach().clone() for k, v in actor.state_dict().items()}
        elif ep - best_epoch >= patience:
            break
    actor.load_state_dict(best_state)
    # El actor objetivo no existe en SAC (solo crítico objetivo); el optimizador del actor de SB3 no se toca aquí.
    return {'best_epoch': best_epoch, 'epochs_run': len(hist), 'best_val_mse': float(best),
            'train_wall_s': round(time.time() - t0, 1),
            'train': evaluate_actor(actor, obs[train_idx], act[train_idx]),
            'val': evaluate_actor(actor, obs[val_idx], act[val_idx]) if len(val_idx) else None}


def load_model(path: str, env=None, device: str = 'cpu', **kw):
    return SAC.load(path, env=env, device=device, **kw)


def load_policy(path: str, deterministic: bool = True):
    """Interfaz documentada: act = load_policy('model.zip'); a = act(obs31) -> np.ndarray (7,) en [-1, 1]."""
    model = load_model(path)

    def act(obs):
        return model.predict(np.asarray(obs, np.float32), deterministic=deterministic)[0]

    act.model = model
    return act


def save_json(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(data, fh, indent=1)
