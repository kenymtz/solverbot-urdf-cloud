"""Almacenamiento y carga de demostraciones.

Estructura de un conjunto (directorio):
    meta.json                  experimento, nivel, origen (expert | dagger_<iter>), config, commit
    index.jsonl                un resumen por episodio (rollout.run_episode) + 'file'
    episodes/ep_<seed>.npz     arrays por paso (obs, actions, expert_actions, rewards, next_obs, terminated, ...)

La entrada de la política es SOLO `obs` (POLICY_INPUT_KEY). Las etiquetas son `expert_actions`. Las claves `priv_*`
(verdad del simulador) se conservan para auditoría y nunca se devuelven como entrada.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .rollout import ACT_DIM, OBS_DIM

POLICY_INPUT_KEY = 'obs'
LABEL_KEY = 'expert_actions'


def episode_file(root: str, seed: int) -> str:
    return os.path.join(root, 'episodes', f'ep_{seed}.npz')


def save_episode(root: str, arrays: dict, summary: dict) -> dict:
    os.makedirs(os.path.join(root, 'episodes'), exist_ok=True)
    path = episode_file(root, summary['seed'])
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    row = {**summary, 'file': os.path.relpath(path, root)}
    with open(os.path.join(root, 'index.jsonl'), 'a') as fh:
        fh.write(json.dumps(row) + '\n')
    return row


def read_index(root: str) -> list[dict]:
    path = os.path.join(root, 'index.jsonl')
    if not os.path.exists(path):
        return []
    rows = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[r['seed']] = r                 # la última entrada de una semilla manda (reintentos)
    return list(rows.values())


def validate_episode(arrays: dict) -> list[str]:
    """Problemas estructurales de un episodio (lista vacía = válido)."""
    errs = []
    n = len(arrays['actions'])
    for k in ('obs', 'next_obs', 'actions', 'expert_actions', 'rewards', 'terminated', 'truncated'):
        if k not in arrays:
            errs.append(f'falta {k}')
        elif len(arrays[k]) != n:
            errs.append(f'{k}: longitud {len(arrays[k])} != {n}')
    if errs:
        return errs
    if arrays['obs'].shape[1:] != (OBS_DIM,) or arrays['next_obs'].shape[1:] != (OBS_DIM,):
        errs.append(f'dimensión de obs {arrays["obs"].shape}')
    if arrays['actions'].shape[1:] != (ACT_DIM,) or arrays['expert_actions'].shape[1:] != (ACT_DIM,):
        errs.append(f'dimensión de acción {arrays["actions"].shape}')
    for k in ('obs', 'next_obs', 'actions', 'expert_actions', 'rewards'):
        if not np.isfinite(arrays[k]).all():
            errs.append(f'{k} contiene NaN/inf')
    if np.abs(arrays['expert_actions']).max(initial=0) > 1.0 + 1e-6:
        errs.append('etiquetas fuera de [-1, 1]')
    if not (bool(arrays['terminated'][-1]) or bool(arrays['truncated'][-1])):
        errs.append('el episodio no termina')
    if np.any(arrays['terminated'][:-1] | arrays['truncated'][:-1]):
        errs.append('fin de episodio antes del último paso')
    # continuidad: la observación siguiente de un paso es la observación del paso siguiente
    if n > 1 and not np.allclose(arrays['next_obs'][:-1], arrays['obs'][1:]):
        errs.append('next_obs[t] != obs[t+1]')
    return errs


@dataclass
class Dataset:
    obs: np.ndarray                 # (N, 31) entrada de la política
    actions: np.ndarray             # (N, 7) etiquetas del experto
    episode: np.ndarray             # (N,) id de episodio (para separar train/val por episodio)
    episodes: list[dict]            # resúmenes incluidos

    def __len__(self):
        return len(self.obs)


def load_datasets(roots: list[str], only_success: bool = True, require_labels: bool = True) -> Dataset:
    """Concatena conjuntos. only_success: en demostraciones puras del experto, descarta episodios fallidos (el
    experto falló: sus etiquetas no muestran cómo resolver). En DAgger (etiquetas del experto sobre estados de la
    política) los episodios fallidos SÍ se usan: ahí está la información de corrección."""
    obs, act, ep, eps = [], [], [], []
    for root in roots:
        meta = {}
        if os.path.exists(os.path.join(root, 'meta.json')):
            with open(os.path.join(root, 'meta.json')) as fh:
                meta = json.load(fh)
        dagger = str(meta.get('source', 'expert')).startswith('dagger')
        for row in sorted(read_index(root), key=lambda r: r['seed']):
            if only_success and not dagger and not row['success']:
                continue
            with np.load(os.path.join(root, row['file'])) as z:
                arrays = {k: z[k] for k in z.files}
            errs = validate_episode(arrays)
            if errs:
                raise ValueError(f'{root}/{row["file"]}: {errs}')
            if require_labels and dagger is False and not np.allclose(arrays['actions'], arrays['expert_actions']):
                raise ValueError(f'{root}/{row["file"]}: demostración del experto con acciones != etiquetas')
            obs.append(arrays[POLICY_INPUT_KEY])
            act.append(arrays[LABEL_KEY])
            ep.append(np.full(len(arrays['obs']), len(eps)))
            eps.append({**row, 'root': root})
    if not obs:
        raise ValueError(f'sin episodios utilizables en {roots}')
    return Dataset(np.concatenate(obs).astype(np.float32), np.concatenate(act).astype(np.float32),
                   np.concatenate(ep), eps)


def split_by_episode(ds: Dataset, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Índices de muestras train/val separando EPISODIOS completos (muestras de un mismo episodio están
    correlacionadas: mezclarlas inflaría la validación)."""
    n_ep = len(ds.episodes)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_ep)
    n_val = max(1, int(round(val_frac * n_ep))) if n_ep > 1 else 0
    val_eps = set(perm[:n_val].tolist())
    is_val = np.isin(ds.episode, list(val_eps))
    return np.nonzero(~is_val)[0], np.nonzero(is_val)[0]
