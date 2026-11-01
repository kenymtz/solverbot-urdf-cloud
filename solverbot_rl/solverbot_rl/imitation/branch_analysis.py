"""Diagnóstico de las ramas de muñeca en `reach` (Fase A de exp/solverbot-wrist-branch-consistency).

SOLO ANÁLISIS: no lo usa ninguna política, experto ni entorno. Usa la verdad del simulador (pose real del cubo) para
calcular las soluciones de IK de referencia de cada episodio; eso es una medición, nunca una entrada de la política.

Estructura de las soluciones (medida, ver docs/WRIST_BRANCHES.md): con el eje de aproximación fijo y el yaw libre,
joint1-joint3 quedan determinados por la inclinación (20/30/40°), joint6 es un continuo libre (giro alrededor del eje
de aproximación) y (joint4, joint5) tiene DOS familias discretas, un "volteo de muñeca" (joint4 ± pi, -joint5):
familia A con joint4 ≈ +1.5 y familia B con joint4 ≈ -1.5. La etiqueta de familia de una solución es el signo de
joint4 (|joint4| ≈ 1.5 en ambas); el signo de joint5 depende de la inclinación.

Cuatro nociones distintas (no se confunden):
  * rama ELEGIDA por el experto: familia de su objetivo articular q_goal (IK global);
  * rama que INTENTA seguir una política: hacia qué familia apunta la orden articular de la muñeca en cada paso;
  * rama EJECUTADA: a qué familia se acerca la configuración medida (afinidad con histéresis);
  * cambio de SIGNO de la acción de joint4: evento de la acción, que NO implica un cambio de rama.
"""
from __future__ import annotations

import numpy as np

from ..kinematics import solve_ik

TILTS = (20, 30, 40)
FAMILIES = ('A', 'B')                    # A: joint4 > 0, B: joint4 < 0 (en la solución objetivo)
J4, J5 = 3, 4
WRIST = slice(3, 5)                      # joint4, joint5 (joint6 es libre y se excluye)
COMMIT = 0.3                             # umbral de afinidad para "comprometido" con una familia (histéresis)
INTENT_COS = 0.5                         # coseno mínimo para atribuir una intención
SIGN_MIN = 0.3                           # |a4| mínimo a ambos lados para contar un cambio de signo


def approach_axis(tilt_deg: float) -> np.ndarray:
    th = np.radians(tilt_deg)
    return np.array([np.sin(th), 0.0, -np.cos(th)])


def family_of(q) -> str:
    return 'A' if q[J4] > 0 else 'B'


def ik_solution_set(env, q0, p, n_starts: int = 24, seed: int = 0) -> list[dict]:
    """Soluciones de IK de pre-agarre (eje de aproximación con yaw libre, como ScriptedExpert._ik) desde n_starts
    arranques por inclinación: el primero en q0 y el resto uniformes en los límites. Solo las convergidas (< 3 mm)."""
    u = env.unwrapped
    fixed = {'torso_lift_joint': u.cfg['robot']['lift']}
    rng = np.random.default_rng(seed)
    out = []
    for tilt in TILTS:
        for k in range(n_starts):
            start = q0 if k == 0 else rng.uniform(u.lo, u.hi)
            q, err = solve_ik(u.chain, 'tcp_link', u.arm_joints, start, p, None, fixed,
                              approach_axis=approach_axis(tilt), rot_weight=1.0, restarts=0, reg=0.002)
            if err < 0.003:
                out.append({'tilt': tilt, 'q': np.asarray(q, float), 'family': family_of(q)})
        # la otra familia se busca explícitamente desde el VOLTEO de muñeca de cada familia hallada (joint4 ∓ pi,
        # -joint5, joint6 ± pi): con pocos arranques aleatorios a veces solo se encuentra una
        for f in FAMILIES:
            if any(s['family'] == f and s['tilt'] == tilt for s in out):
                continue
            for s in [s for s in out if s['tilt'] == tilt][:4]:
                flip = s['q'].copy()
                flip[J4] -= np.sign(flip[J4]) * np.pi
                flip[J5] = -flip[J5]
                flip[5] -= np.sign(flip[5]) * np.pi
                q, err = solve_ik(u.chain, 'tcp_link', u.arm_joints, np.clip(flip, u.lo, u.hi), p, None, fixed,
                                  approach_axis=approach_axis(tilt), rot_weight=1.0, restarts=0, reg=0.002)
                if err < 0.003 and family_of(q) == f:
                    out.append({'tilt': tilt, 'q': np.asarray(q, float), 'family': f})
                    break
    return out


def family_costs(sols: list[dict], q0, delta: float, tilt: int | None = None) -> dict:
    """Por familia, la solución de menor coste desde q0. Coste L2 (el criterio de desempate de la IK del experto,
    dominado por la regularización hacia q0) y pasos mínimos = max|dq| / delta (velocidad articular acotada).
    tilt: restringir a una inclinación (None = la primera que tenga solución, como el experto)."""
    q0 = np.asarray(q0, float)
    if tilt is None:
        tilts = [t for t in TILTS if any(s['tilt'] == t for s in sols)]
        tilt = tilts[0] if tilts else None
    out = {'tilt': tilt}
    for f in FAMILIES:
        cand = [s['q'] for s in sols if s['family'] == f and s['tilt'] == tilt]
        if not cand:
            out[f] = None
            continue
        best = min(cand, key=lambda q: float(np.linalg.norm(q - q0)))
        dq = best - q0
        out[f] = {'q': best, 'l2': float(np.linalg.norm(dq)), 'l2_wrist': float(np.linalg.norm(dq[WRIST])),
                  'steps': float(np.max(np.abs(dq)) / delta), 'dj4': float(dq[J4]), 'dj5': float(dq[J5])}
    return out


def affinity(q, rep_a, rep_b) -> float:
    """+1 = la muñeca medida está sobre la solución A, -1 = sobre B, 0 = equidistante (p. ej. en el arranque)."""
    da = float(np.linalg.norm(np.asarray(q)[WRIST] - rep_a[WRIST]))
    db = float(np.linalg.norm(np.asarray(q)[WRIST] - rep_b[WRIST]))
    return (db - da) / max(da + db, 1e-9)


def executed_branch(aff: np.ndarray, commit: float = COMMIT) -> tuple[list, list[int]]:
    """Familia ejecutada por paso con histéresis: se compromete con A si afinidad > commit, con B si < -commit, y
    conserva la última mientras está en la banda intermedia. Devuelve (etiquetas, pasos de CAMBIO REAL de rama:
    paso de comprometido con una familia a comprometido con la otra)."""
    lab, cur, changes = [], None, []
    for t, s in enumerate(aff):
        new = 'A' if s > commit else 'B' if s < -commit else cur
        if cur is not None and new != cur:
            changes.append(t)
        cur = new
        lab.append(cur)
    return lab, changes


def intended_branch(q, a, rep_a, rep_b, cos_min: float = INTENT_COS) -> str | None:
    """Familia hacia la que apunta la orden articular de la muñeca (a[3:5]): coseno con (rep_f - q)[3:5]. None si
    la orden es casi nula o no apunta claramente a ninguna (intención ambigua)."""
    v = np.asarray(a, float)[WRIST]
    if np.linalg.norm(v) < 1e-3:
        return None
    best, best_c = None, cos_min
    for f, rep in (('A', rep_a), ('B', rep_b)):
        d = rep[WRIST] - np.asarray(q, float)[WRIST]
        n = np.linalg.norm(d)
        if n < 0.05:                          # ya sobre esa solución: no se puede inferir intención
            continue
        c = float(v @ d / (np.linalg.norm(v) * n))
        if c > best_c:
            best, best_c = f, c
    return best


def sign_changes(a4: np.ndarray, min_mag: float = SIGN_MIN) -> list[int]:
    """Cambios de signo de la acción de joint4 con |a4| >= min_mag a ambos lados (los ceros intermedios se saltan)."""
    out, last = [], None
    for t, v in enumerate(np.asarray(a4, float)):
        if abs(v) < min_mag:
            continue
        s = np.sign(v)
        if last is not None and s != last:
            out.append(t)
        last = s
    return out


def intention_switches(intent: list, persist: int = 3, active=None) -> list[int]:
    """Cambios de la familia hacia la que apunta la orden de la muñeca, con REBOTE: la nueva intención tiene que
    sostenerse `persist` pasos consecutivos (no nulos). active: máscara de pasos considerados (p. ej. lejos del
    objetivo, donde la orden decide la rama; cerca, los ajustes finos no indican rama)."""
    out, cur, run_f, run_n, run_t = [], None, None, 0, None
    for t, f in enumerate(intent):
        if f is None or (active is not None and not active[t]):
            continue
        if f == run_f:
            run_n += 1
        else:
            run_f, run_n, run_t = f, 1, t
        if run_n >= persist and run_f != cur:
            if cur is not None:
                out.append(run_t)
            cur = run_f
    return out


def detour_mm(dist) -> float:
    """Máxima subida de la distancia al objetivo respecto de su mínimo previo (mm): costo de trayectoria de una
    indecisión (0 en una aproximación monótona)."""
    d = np.asarray(dist, float)
    return float(1000 * np.max(d - np.minimum.accumulate(d))) if len(d) else 0.0


def wrist_limit_frac(q, lo, hi, tol: float = 0.02) -> float:
    """Fracción de pasos con joint4 o joint5 a menos de tol de su límite articular."""
    q = np.asarray(q, float)
    w = q[:, 3:5]
    return float(np.mean(np.any((w <= lo[3:5] + tol) | (w >= hi[3:5] - tol), axis=1)))


def actor_features(obs: np.ndarray) -> np.ndarray:
    """Entrada efectiva del actor BC (imitation/bc.py enmascara la acción previa): obs sin prev_action."""
    obs = np.asarray(obs, float)
    return np.concatenate([obs[..., :20], obs[..., 27:31]], axis=-1)


def denorm_q(obs: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return lo + (np.asarray(obs, float)[..., :6] + 1.0) * (hi - lo) / 2.0


def label_conflicts(X: np.ndarray, fam: np.ndarray, ep: np.ndarray, a4: np.ndarray, radius: float,
                    chunk: int = 2048) -> dict:
    """Pares de pasos de EPISODIOS DISTINTOS con entrada del actor a distancia < radius (espacio estandarizado).
    Cuenta los pares con familia opuesta y compara la diferencia de acción de joint4 en pares en conflicto frente a
    pares de la misma familia. X ya estandarizado."""
    n = len(X)
    conf = same = 0
    da_conf, da_same = [], []
    has_conf = np.zeros(n, bool)
    for i0 in range(0, n, chunk):
        xi = X[i0:i0 + chunk]
        d2 = (xi ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None, :] - 2 * xi @ X.T
        near = (d2 < radius ** 2) & (ep[i0:i0 + chunk, None] != ep[None, :])
        ii, jj = np.nonzero(near)
        ii = ii + i0
        keep = ii < jj
        ii, jj = ii[keep], jj[keep]
        opp = fam[ii] != fam[jj]
        conf += int(opp.sum())
        same += int((~opp).sum())
        da_conf.extend(np.abs(a4[ii[opp]] - a4[jj[opp]]).tolist())
        da_same.extend(np.abs(a4[ii[~opp]] - a4[jj[~opp]]).tolist())
        has_conf[ii[opp]] = True
        has_conf[jj[opp]] = True
    return {'pairs_conflict': conf, 'pairs_same': same,
            'conflict_rate': conf / max(conf + same, 1),
            'da4_conflict_mean': float(np.mean(da_conf)) if da_conf else None,
            'da4_same_mean': float(np.mean(da_same)) if da_same else None,
            'steps_in_conflict': int(has_conf.sum()), 'has_conflict': has_conf}
