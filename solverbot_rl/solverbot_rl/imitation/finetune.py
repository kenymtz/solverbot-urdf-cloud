"""Fine-tuning SAC a partir de la política clonada.

Riesgo principal: el crítico de un SAC recién inicializado es aleatorio; si el actor BC se actualiza contra él desde el
primer paso, el gradiente de Q destruye la política clonada antes de que el crítico valga algo. Además:
  * SB3 usa acciones ALEATORIAS uniformes mientras num_timesteps < learning_starts -> aquí learning_starts = 0 y el
    calentamiento se hace con `actor_freeze_updates` (el BC actúa desde el primer paso).
  * El coeficiente de entropía guardado en un modelo nuevo vale 1.0 (log 0), enorme frente a la escala de las
    recompensas de `reach` (~0.01-0.1 por paso + bono 10): se fija explícitamente (set_entropy).

`SACBC` = SAC de SB3 2.x con dos agregados en train():
  1. actor_freeze_updates: las primeras N actualizaciones solo entrenan el crítico (evaluación de la política BC).
  2. bc_coef > 0: pérdida del actor += bc_coef * MSE(tanh(mu(s_demo)), a_demo) (regularización hacia el experto,
     estilo TD3+BC / AWAC); 0 = SAC puro.
"""
from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


class SACBC(SAC):
    actor_freeze_updates: int = 0
    bc_coef: float = 0.0
    q_alpha: float = 0.0              # > 0: normalización TD3+BC del término RL del actor
    actor_lr_scale: float = 1.0       # tasa del actor = learning_rate * actor_lr_scale (crítico: learning_rate)
    _bc_obs = None
    _bc_act = None

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ['_bc_obs', '_bc_act']

    def _update_learning_rate(self, optimizers):
        super()._update_learning_rate(optimizers)
        for g in self.actor.optimizer.param_groups:
            g['lr'] = self.lr_schedule(self._current_progress_remaining) * self.actor_lr_scale

    def set_bc_data(self, obs: np.ndarray, act: np.ndarray):
        self._bc_obs = th.as_tensor(obs, dtype=th.float32, device=self.device)
        self._bc_act = th.as_tensor(act, dtype=th.float32, device=self.device)

    def set_entropy(self, ent_coef: float, learn: bool, target_entropy: float | None = None):
        """Fija alpha. learn=True: alpha automático partiendo de ent_coef con objetivo target_entropy."""
        if learn:
            if self.log_ent_coef is None:
                raise ValueError('el modelo se creó con ent_coef fijo; no se puede volver automático sin recrearlo')
            with th.no_grad():
                self.log_ent_coef.fill_(float(np.log(ent_coef)))
            if target_entropy is not None:
                self.target_entropy = float(target_entropy)
        else:
            self.ent_coef = float(ent_coef)     # se guarda como fijo: al cargar, _setup_model crea ent_coef_tensor
            self.ent_coef_optimizer = None
            self.log_ent_coef = None
            self.ent_coef_tensor = th.tensor(float(ent_coef), device=self.device)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:  # noqa: C901 (copia anotada de SAC.train)
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)
        ent_coef_losses, ent_coefs, actor_losses, critic_losses, bc_losses = [], [], [], [], []

        for gradient_step in range(gradient_steps):
            frozen = self._n_updates + gradient_step < self.actor_freeze_updates
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            if self.use_sde:
                self.actor.reset_noise()
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None and not frozen:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if not frozen:
                q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                if self.q_alpha > 0:
                    # TD3+BC: el término RL se escala por alpha / mean|Q| (independiente de la escala de recompensa)
                    actor_loss = actor_loss * (self.q_alpha / min_qf_pi.abs().mean().detach().clamp_min(1e-3))
                if self.bc_coef > 0 and self._bc_obs is not None:
                    idx = th.randint(0, len(self._bc_obs), (batch_size,), device=self.device)
                    mu, _, _ = self.actor.get_action_dist_params(self._bc_obs[idx])
                    bc = F.mse_loss(th.tanh(mu), self._bc_act[idx])
                    bc_losses.append(bc.item())
                    actor_loss = actor_loss + self.bc_coef * bc
                actor_losses.append(actor_loss.item())
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        self.logger.record('train/ent_coef', np.mean(ent_coefs))
        self.logger.record('train/critic_loss', np.mean(critic_losses))
        self.logger.record('train/actor_frozen', float(self._n_updates <= self.actor_freeze_updates))
        if actor_losses:
            self.logger.record('train/actor_loss', np.mean(actor_losses))
        if bc_losses:
            self.logger.record('train/bc_loss', np.mean(bc_losses))
        if ent_coef_losses:
            self.logger.record('train/ent_coef_loss', np.mean(ent_coef_losses))


def add_demos_to_buffer(model, obs, next_obs, actions, rewards, terminated, truncated) -> int:
    """Agrega transiciones (de demostraciones/DAgger) al replay buffer con el mismo tratamiento de truncamiento que
    DummyVecEnv (TimeLimit.truncated => no es terminal para el bootstrap)."""
    rb = model.replay_buffer
    for i in range(len(obs)):
        done = bool(terminated[i] or truncated[i])
        info = {'TimeLimit.truncated': bool(truncated[i] and not terminated[i])}
        rb.add(obs[i][None], next_obs[i][None], actions[i][None], np.array([rewards[i]], np.float32),
               np.array([done]), [info])
    return len(obs)


def actor_param_vector(model) -> np.ndarray:
    return th.cat([p.detach().flatten().cpu() for p in model.actor.parameters()]).numpy().copy()


@th.no_grad()
def policy_entropy_estimate(model, obs: np.ndarray, n: int = 2048) -> float:
    """-E[log pi(a|s)] de la política estocástica sobre estados de demostración (para fijar el objetivo de entropía)."""
    o = th.as_tensor(obs[:n], dtype=th.float32, device=model.device)
    _, log_prob = model.actor.action_log_prob(o)
    return float(-log_prob.mean())
