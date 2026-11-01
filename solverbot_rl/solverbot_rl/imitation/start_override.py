"""Arranques con joint4 girado (exp/solverbot-wrist-branch-consistency, B2/B3). SOLO backend cinemático.

Misma semilla (cubo, yaw del cubo y demás articulaciones) pero joint4 inicial fijado en un valor o sorteado en un
rango con la RNG de la semilla. Sirve para:
  * evaluar arranques próximos a la frontera de compromiso entre familias de muñeca (valor fijo);
  * dar cobertura de esos estados a demostraciones y DAgger en semillas de ENTRENAMIENTO (rango).
No modifica el entorno, sus niveles, la recompensa ni el criterio de éxito; la política sigue viendo solo la
observación de 31 dimensiones.
"""
from __future__ import annotations

import numpy as np

from .. import task as T


def override_start_j4(env, value: float | None = None, low: float | None = None, high: float | None = None):
    """Envuelve env.reset: después del reset original, reubica el brazo con joint4 = value (o uniforme en
    [low, high] con una RNG derivada de la semilla del episodio)."""
    if env.unwrapped.backend.physical:
        raise ValueError('override_start_j4: solo en el backend cinemático')
    if value is None and (low is None or high is None):
        raise ValueError('valor fijo o rango [low, high]')
    orig = env.reset

    def reset(*a, **k):
        obs, info = orig(*a, **k)
        u = env.unwrapped
        q = np.array(u.start_q, float)
        if value is not None:
            q[3] = value
        else:
            seed = k.get('seed')
            rng = np.random.default_rng(None if seed is None else [int(seed), 4])
            q[3] = rng.uniform(low, high)
        q = np.clip(q, u.lo + 0.02, u.hi - 0.02)
        b = u.backend
        r = u.cfg['robot']
        b.reset(b.rng, u.start_obj_xy, b.obj_yaw, q, r['gripper_open'], r['lift'], r['head'])
        u.start_q = q
        u._q_cmd = q.copy()
        u._state = u._task_state(u._joints())
        u.flags.best_phi = T.potential(u._state, u.params)
        info = dict(info)
        info['start_j4_override'] = float(q[3])
        return u._obs(u._state), info

    env.reset = reset
    return env


def override_start_cube_home(env, holder: dict):
    """Envuelve env.reset (Fase C, verificación de la grilla de B0): si holder['dxy'] no es None, reubica el cubo en
    la posición nominal del mundo + dxy y el brazo en `home` exacto, como la grilla de factibilidad. SOLO cinemático;
    no cambia niveles, recompensa ni criterio de éxito. holder es mutable para cambiar de celda entre episodios."""
    if env.unwrapped.backend.physical:
        raise ValueError('override_start_cube_home: solo en el backend cinemático')
    orig = env.reset

    def reset(*a, **k):
        obs, info = orig(*a, **k)
        if holder.get('dxy') is None:
            return obs, info
        u = env.unwrapped
        q = np.asarray(u.cfg['robot']['home'], float)
        xy = np.asarray(u.cfg['world']['object']['xy'], float) + np.asarray(holder['dxy'], float)
        b = u.backend
        r = u.cfg['robot']
        b.reset(b.rng, xy, b.obj_yaw, q, r['gripper_open'], r['lift'], r['head'])
        u.start_q, u.start_obj_xy = q, xy
        u._q_cmd = q.copy()
        u._state = u._task_state(u._joints())
        u.flags.best_phi = T.potential(u._state, u.params)
        info = dict(info)
        info['start_cube_dxy'] = [float(v) for v in holder['dxy']]
        return u._obs(u._state), info

    env.reset = reset
    return env
