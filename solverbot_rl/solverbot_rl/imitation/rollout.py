"""Ejecución de episodios con registro completo (demostraciones, DAgger y evaluación autónoma).

Separación estricta:
  * `obs` (31) es lo ÚNICO que ve la política: lo entrega env.step/env.reset tal cual (object_source del entorno).
  * Las magnitudes `priv_*` (pose real del cubo y del TCP), las distancias y el error de percepción salen del
    simulador y se usan solo para etiquetar, verificar y medir.
En evaluación autónoma `expert` es None: el experto no se instancia ni se consulta.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from .. import task as T

# Disposición de la observación (envs/pick_place.py::_obs)
OBS_SLICES = {
    'q': slice(0, 6), 'gripper': slice(6, 7), 'tcp': slice(7, 10), 'tcp_z': slice(10, 13),
    'obj_rel': slice(13, 16), 'obj_valid': slice(16, 17), 'target_rel': slice(17, 20),
    'prev_action': slice(20, 27), 'goal': slice(27, 31),
}
OBS_DIM = 31
ACT_DIM = 7


def _metrics(env):
    u = env.unwrapped
    s = u._state
    d = float(np.linalg.norm(s.tcp_pos - T.approach_point(s, u.params)))
    return d, float(s.tcp_z[2]), np.array(s.obj_pos, float), np.array(s.tcp_pos, float)


def run_episode(env, seed: int, level: int, act=None, expert=None, beta: float = 1.0,
                rng: np.random.Generator | None = None) -> tuple[dict, dict]:
    """Un episodio. act(obs) -> acción de la política (None = ejecutar el experto). expert: ScriptedExpert (o None).
    beta: probabilidad de EJECUTAR la acción del experto en cada paso cuando también hay política (DAgger: beta=0
    ejecuta la política y el experto solo etiqueta). Devuelve (arrays por paso, resumen del episodio)."""
    if act is None and expert is None:
        raise ValueError('se necesita una política o un experto')
    rng = rng or np.random.default_rng(seed)
    t_wall = time.time()
    obs, info = env.reset(seed=seed, options={'level': level})
    u = env.unwrapped
    if expert is not None:
        expert.reset()
    rec: dict[str, list] = defaultdict(list)
    d0, z0, obj0, _ = _metrics(env)
    done = False
    info = {}
    while not done:
        label = expert.act() if expert is not None else None
        if act is None:
            a, from_expert = label, True
        else:
            from_expert = expert is not None and rng.random() < beta
            a = label if from_expert else np.asarray(act(obs), np.float32)
        a = np.clip(np.asarray(a, np.float32), -1.0, 1.0)
        _, _, _, tcp_before = _metrics(env)
        obs2, r, term, trunc, info = env.step(a)
        d, z, obj, tcp = _metrics(env)
        est = obs[OBS_SLICES['obj_rel']] + obs[OBS_SLICES['tcp']]     # estimación que vio la política
        valid = bool(obs[16] > 0.5)
        truth_before = rec['priv_obj'][-1] if rec['priv_obj'] else obj0
        rec['obs'].append(np.asarray(obs, np.float32))
        rec['actions'].append(a)
        rec['expert_actions'].append(np.zeros(ACT_DIM, np.float32) if label is None else np.asarray(label, np.float32))
        rec['exec_expert'].append(from_expert)
        rec['rewards'].append(float(r))
        rec['next_obs'].append(np.asarray(obs2, np.float32))
        rec['terminated'].append(bool(term))
        rec['truncated'].append(bool(trunc))
        rec['dist'].append(d)
        rec['tcp_z2'].append(z)
        rec['obj_valid'].append(valid)
        rec['perc_err'].append(float(np.linalg.norm(est - truth_before)) if valid else np.nan)
        rec['priv_obj'].append(obj)
        rec['priv_tcp'].append(tcp)
        rec['sim_time'].append(float(getattr(u.backend, '_clock', np.nan)))
        rec['stale_images'].append(int(info.get('stale_images', 0)))
        obs = obs2
        done = term or trunc
    arrays = {k: np.asarray(v) for k, v in rec.items()}
    arrays['obs'] = arrays['obs'].astype(np.float32)
    arrays['next_obs'] = arrays['next_obs'].astype(np.float32)
    arrays['actions'] = arrays['actions'].astype(np.float32)
    arrays['expert_actions'] = arrays['expert_actions'].astype(np.float32)
    es = info['episode_summary']
    pe = arrays['perc_err'][np.isfinite(arrays['perc_err'])]
    summary = {
        'seed': int(seed), 'level': int(level), 'level_name': u.levels[level]['name'], 'goal': u.params.goal,
        'object_source': u.object_source, 'physical': bool(u.backend.physical),
        'obj_xy_cmd': [float(v) for v in u.start_obj_xy], 'start_q': [float(v) for v in u.start_q],
        'obj_start_truth': [float(v) for v in obj0],
        'success': bool(es['success']), 'return': float(es['return']), 'length': int(es['length']),
        'stalled': bool(es['stalled']), 'collision': 'pen_collision' in es['terms'], 'drop': 'pen_drop' in es['terms'],
        'terms': es['terms'],
        'd0_mm': 1000 * d0, 'dmin_mm': 1000 * float(arrays['dist'].min()), 'dfinal_mm': 1000 * float(arrays['dist'][-1]),
        'tcp_z2_start': z0, 'tcp_z2_final': float(arrays['tcp_z2'][-1]),
        'valid_rate': float(arrays['obj_valid'].mean()), 'n_invalid': int((~arrays['obj_valid']).sum()),
        'perc_err_mean_mm': 1000 * float(pe.mean()) if pe.size else None,
        'perc_err_max_mm': 1000 * float(pe.max()) if pe.size else None,
        'stale_images_total': int(arrays['stale_images'].sum()),
        'reset_error': float(getattr(u.backend, 'reset_error', 0.0)),
        'expert_exec_rate': float(arrays['exec_expert'].mean()),
        'has_nan': bool(not (np.isfinite(arrays['obs']).all() and np.isfinite(arrays['actions']).all()
                             and np.isfinite(arrays['next_obs']).all())),
        'dims_ok': bool(arrays['obs'].shape[1:] == (OBS_DIM,) and arrays['actions'].shape[1:] == (ACT_DIM,)),
        'wall_s': round(time.time() - t_wall, 2), 'time': time.time(),
    }
    return arrays, summary
