"""Etiquetador de IK RECALCULADA EN CADA PASO desde el estado articular medido (Fase C de
exp/solverbot-wrist-branch-consistency).

OPCIONAL y experimental: solo se usa con `--labeler dynamic` / `--policy dynamic`. No reemplaza ni modifica
ScriptedExpert (experto por defecto) ni BranchConsistentLabelExpert (Fase B). No cambia el entorno, la física, la
recompensa, los límites articulares ni el criterio de éxito.

Motivo (docs/WRIST_BRANCHES_FASE_B.md): el etiquetador de la Fase B fija `q_goal` en la primera decisión del episodio
y lo reutiliza; joint6 es libre y su valor depende del arranque, así que dos estados observados iguales recibían
objetivos distintos, y DAgger (bc_d2) empeoró. Los fallos dominantes de los BC de la Fase B eran saturaciones de
joint4/joint5.

Regla, evaluada en CADA paso de la fase de pre-agarre de `reach`:
  1. Candidatas: soluciones de IK de pre-agarre (eje de aproximación con yaw libre, 20° y 30°; 40° nunca cumple el
     criterio de éxito) obtenidas por optimización LOCAL desde la articulación medida q, desde su volteo de muñeca y,
     si falta una familia, desde semillas canónicas de esa familia, y desde la solución del paso anterior de cada
     clave (solo como semilla: así una solución ya hallada no desaparece por depender del arranque del optimizador).
     Se agrupan por clave (familia, inclinación) y se elige la de menor coste.
  2. joint6 del objetivo = joint6 medido (recortado a sus límites reales con margen). joint6 no mueve el TCP ni el
     eje de aproximación (verificado), así que la etiqueta no lo gira: continuidad articular y sin dependencia del
     arranque.
  3. Coste de selección (NO es la recompensa de Gymnasium): distancia articular desde q (joint1..joint5) +
     W_LIMIT * sum_{j4, j5} ((LIMIT_SOFT - margen)+ / LIMIT_SOFT)^2.
  4. Validez de cada candidata en el paso: IK < 3 mm, orientación de éxito (tcp_z[2] < -0.8), joint4/joint5 a más de
     LIMIT_MARGIN de sus límites, cubo visible en la cámara de muñeca en el objetivo; y, para la candidata elegida,
     camino del experto sin colisión con el cuerpo, autocolisión, mesa ni muñeca en el límite (branch_feasibility).
  5. Familia: B si la muñeca medida no está comprometida con A (afinidad > COMMIT respecto de las mejores candidatas
     de cada familia), A si lo está (continuidad: nunca fuerza el regreso). Si la familia elegida no tiene candidata
     válida, la otra (respaldo de familia, registrado). Banda de histéresis FAMILY_HYST para que el ruido articular
     no alterne la familia en la frontera de compromiso.
  6. Inclinación: la de menor coste dentro de la familia; la anterior se conserva salvo mejora > TILT_HYST.
  7. Muñeca singular (|joint5| < SING): A y B son el mismo continuo de soluciones (familia neutra 'S', compatible
     con ambas); se toma el punto del continuo más cercano a la articulación medida (project_closest, SLSQP con el
     TCP en el objetivo y el eje dentro de AXIS_TOL).
  8. Refinamiento de límites de la elegida: mismo TCP, eje dentro de AXIS_TOL (~3,4°), minimizando la distancia a
     la solución sin refinar + la penalización suave de joint4/joint5. Se aplica siempre que el resultado sea
     válido (aplicarlo de forma intermitente producía saltos de objetivo).
  9. Etiqueta: paso articular acotado hacia el objetivo del paso, clip((q_goal - q_cmd) / delta).
 10. Sin candidata válida: IK global de ScriptedExpert (respaldo explícito, registrado).

La memoria entre pasos se limita a (a) la familia e inclinación del paso anterior, usadas SOLO dentro de las bandas de
histéresis, (b) las soluciones del paso anterior como semillas adicionales y (c) un caché de la verificación de camino
de la clave elegida. El objetivo en sí es función del estado
medido. Usa la pose verdadera del cubo como ScriptedExpert: es un etiquetador, no una política; la política
entrenada ve solo la observación de 31 dimensiones.
"""
from __future__ import annotations

import numpy as np

from ..expert import ScriptedExpert
from ..kinematics import solve_ik
from . import branch_analysis as BA
from . import branch_feasibility as F

TILTS = (20, 30)
LIMIT_SOFT = 0.35          # rad: margen a partir del cual joint4/joint5 empiezan a penalizar
W_LIMIT = 1.0              # rad equivalentes de distancia por unidad de penalización
J6_MARGIN = 0.10           # rad: margen de joint6 a sus límites reales
TILT_HYST = 0.15           # rad: mejora mínima de coste para cambiar de inclinación
FAMILY_HYST = 0.10         # banda de afinidad para abandonar la familia actual
IK_TOL = 0.003             # m
PATH_RECHECK = 0.30        # rad: deriva del objetivo que obliga a verificar de nuevo el camino
CANON_J4 = {'A': 1.3, 'B': -1.3}
AXIS_TOL = 0.06            # holgura del eje de aproximación en la proyección singular (~3,4°)
SING = 0.12                # rad: |joint5| por debajo del cual la muñeca es singular (familia neutra 'S')


def flip(q):
    """Volteo de muñeca: joint4 ∓ pi, -joint5 (joint6 es libre y se fija aparte)."""
    m = np.asarray(q, float).copy()
    m[3] -= np.sign(m[3] if m[3] != 0 else 1.0) * np.pi
    m[4] = -m[4]
    return m


class MeasuredIKLabelExpert(ScriptedExpert):
    def __init__(self, env, geometry: F.RobotGeometry | None = None, w_limit: float = W_LIMIT,
                 limit_soft: float = LIMIT_SOFT, check_path: bool = True, refine: bool = True):
        self.geom = geometry or F.RobotGeometry()
        self.w_limit, self.limit_soft, self.check_path, self.refine = w_limit, limit_soft, check_path, refine
        super().__init__(env)

    def reset(self):
        super().reset()
        self.family = None
        self.tilt = None
        self.decision = None
        self.viability = None
        self.decisions: list[dict] = []
        self._table = None
        self._path_cache: dict = {}
        self._prev_sol: dict = {}       # última solución por clave: SOLO semilla extra (la elección es por coste)

    # ------------------------------------------------------------------------------------------------ helpers
    def _q(self):
        e = self.env
        return np.array([e._joints()[j] for j in e.arm_joints], float)

    def _full(self, q):
        return {**dict(zip(self.env.arm_joints, q)), 'torso_lift_joint': self.env.cfg['robot']['lift']}

    def _solve(self, seed, p, tilt):
        e = self.env
        q, err = solve_ik(e.chain, 'tcp_link', e.arm_joints, np.clip(seed, e.lo, e.hi), p, None, self._fixed(),
                          approach_axis=BA.approach_axis(tilt), rot_weight=1.0, restarts=0, reg=0.002)
        return np.asarray(q, float), float(err)

    def project_closest(self, sol, q, p, tilt, limits: bool = False):
        """Muñeca singular: la solución MÁS CERCANA a la articulación medida (joint1..joint5) con el TCP en el
        objetivo (igualdad) y el eje de aproximación a menos de AXIS_TOL de la inclinación nominal (desigualdad).
        Junto a |joint5| ~ 0, joint4 y joint6 son casi colineales y el eje exacto fija joint4 en valores alejados
        del medido; la holgura AXIS_TOL (~3,4°, muy dentro del criterio de éxito de 36,9°) une esas soluciones en un
        continuo y permite elegir la más cercana. SLSQP desde `sol` y desde `sol` con joint4 medido y joint5 = 0;
        joint6 fijo en el medido (no afecta al TCP ni al eje)."""
        from scipy.optimize import minimize
        e = self.env
        ax = BA.approach_axis(tilt)
        j6 = float(q[5])

        def pose(x):
            return self.geom.chain.link_pose('tcp_link', self._full(np.append(x, j6)))

        cons = [{'type': 'eq', 'fun': lambda x: (pose(x)[:3, 3] - p) * 10.0},
                {'type': 'ineq', 'fun': lambda x: AXIS_TOL ** 2 - float(np.sum((pose(x)[:3, 2] - ax) ** 2))}]
        lo, hi = e.lo[:5] + 1e-4, e.hi[:5] - 1e-4

        # referencia: el estado medido para la proyección singular (elegir el punto del continuo más cercano); la
        # solución de IK sin refinar para el refinamiento de límites (medir la distancia a q dentro del cono de
        # holgura hace deslizar el óptimo por el borde del cono a medida que q avanza: saltos de 0,15-0,4 rad)
        ref = np.asarray(sol, float)[:5] if limits else q[:5]

        def objective(x):
            d = float(np.sum((x - ref) ** 2))
            if limits:
                m = np.minimum(x - e.lo[:5], e.hi[:5] - x)[3:5]
                d += self.w_limit * float(np.sum((np.clip(self.limit_soft - m, 0.0, None) / self.limit_soft) ** 2))
            return d

        best = (np.asarray(sol, float), 1.0)
        starts = [np.clip(sol[:5], lo, hi)]
        alt = np.array(sol[:5], float)
        alt[3], alt[4] = q[3], 0.0
        starts.append(np.clip(alt, lo, hi))
        for x0 in starts:
            r = minimize(objective, x0, method='SLSQP', bounds=list(zip(lo, hi)),
                         constraints=cons, options={'maxiter': 150, 'ftol': 1e-10})
            t = pose(r.x)
            err = float(np.linalg.norm(t[:3, 3] - p))
            ok = err < IK_TOL and float(np.sum((t[:3, 2] - ax) ** 2)) <= (AXIS_TOL * 1.05) ** 2
            if ok and (best[1] >= IK_TOL or objective(r.x) < objective(best[0][:5])):
                best = (np.append(r.x, j6), err)
        return best

    @staticmethod
    def family_key(qg) -> str:
        """'A' / 'B' por el signo de joint4; 'S' si la muñeca es singular (|joint5| < SING): ahí A y B son el mismo
        continuo de soluciones y el signo de joint4 no distingue ramas."""
        return 'S' if abs(qg[4]) < SING else BA.family_of(qg)

    def limit_penalty(self, qg) -> float:
        e = self.env
        m = np.minimum(qg - e.lo, e.hi - qg)[3:5]
        return float(np.sum((np.clip(self.limit_soft - m, 0.0, None) / self.limit_soft) ** 2))

    def cost(self, q, qg) -> float:
        return float(np.linalg.norm((qg - q)[:5])) + self.w_limit * self.limit_penalty(qg)

    def candidates(self, q, p) -> dict:
        """Mejor candidata (menor coste) por (familia, inclinación), con su validez local. Función de (q, p)."""
        e = self.env
        cube = p - np.array([0.0, 0.0, e.params.pregrasp_height])
        j6 = float(np.clip(q[5], e.lo[5] + J6_MARGIN, e.hi[5] - J6_MARGIN))
        out = {}
        for tilt in TILTS:
            found = []
            seeds = [q, flip(q)] + [v for (f, t), v in self._prev_sol.items() if t == tilt]
            for seed in seeds:
                sol, err = self._solve(seed, p, tilt)
                found.append((sol, err))
            fams = {self.family_key(s) for s, err in found if err < IK_TOL}
            for f in BA.FAMILIES:
                if f in fams:
                    continue
                seed = q.copy()
                seed[3] = CANON_J4[f]
                seed[4] = -abs(q[4]) if f == 'B' else abs(q[4])
                found.append(self._solve(seed, p, tilt))
            # muñeca singular (|joint5| < SING): joint4 y joint6 son casi colineales y joint4 queda en un continuo de
            # soluciones; se busca la del continuo más cercana al joint4 MEDIDO (continuidad, sin depender de la semilla)
            for sol, err in list(found):
                if err < IK_TOL and abs(sol[4]) < SING:
                    found.append(self.project_closest(sol, q, p, tilt))
            for sol, err in found:
                if err >= IK_TOL:
                    continue
                g = sol.copy()
                g[5] = j6
                key = (self.family_key(g), tilt)
                c = self.cost(q, g)
                full = self._full(g)
                tcp = self.geom.chain.link_pose('tcp_link', full)
                m = np.minimum(g - e.lo, e.hi - g)
                reasons = []
                if np.linalg.norm(tcp[:3, 3] - p) > IK_TOL:
                    reasons.append('ik_impreciso')
                if not tcp[2, 2] < F.SUCCESS_TCP_Z:
                    reasons.append('inclinacion_fuera_de_exito')
                if m[3] <= F.LIMIT_MARGIN or m[4] <= F.LIMIT_MARGIN:
                    reasons.append('muneca_en_limite')
                if not F.cube_in_view(F.wrist_camera_pose(self.geom.chain, full), cube):
                    reasons.append('cubo_fuera_de_vista')
                cand = {'q': g, 'cost': c, 'valid': not reasons, 'reason': ','.join(reasons) or None,
                        'limit_pen': self.limit_penalty(g)}
                old = out.get(key)
                if old is None or (not old['valid'], old['cost']) > (not cand['valid'], c):
                    out[key] = cand
                    self._prev_sol[key] = sol
        return out

    def _goal_ok(self, g, p) -> bool:
        e = self.env
        full = self._full(g)
        tcp = self.geom.chain.link_pose('tcp_link', full)
        m = np.minimum(g - e.lo, e.hi - g)
        cube = p - np.array([0.0, 0.0, e.params.pregrasp_height])
        return bool(np.linalg.norm(tcp[:3, 3] - p) <= IK_TOL and tcp[2, 2] < F.SUCCESS_TCP_Z
                    and m[3] > F.LIMIT_MARGIN and m[4] > F.LIMIT_MARGIN
                    and F.cube_in_view(F.wrist_camera_pose(self.geom.chain, full), cube))

    def path_check(self, q, key, qg, p) -> tuple[bool, str | None]:
        """Camino del experto desde q hasta qg: cuerpo, autocolisión, mesa y muñeca libre (branch_feasibility).
        Caché por clave; se verifica de nuevo si el objetivo derivó más de PATH_RECHECK."""
        if not self.check_path:
            return True, None
        c = self._path_cache.get(key)
        if c is not None and np.linalg.norm(c['qg'] - qg) < PATH_RECHECK:
            return c['ok'], c['reason']
        e = self.env
        if self._table is None:
            self._table = F.table_in_base(e)
        x0, x1, y0, y1, ztop = self._table
        path = F.expert_path(q, qg, e.delta)
        reasons = set()
        for qq in path[1::2] if len(path) > 2 else path:
            f = self._full(qq)
            pts = self.geom.link_points(f)
            if self.geom.body_collision(pts, q_full=f):
                reasons.add('colision_cuerpo')
            if self.geom.self_collision(pts):
                reasons.add('autocolision')
            for P in pts.values():
                if ((P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1) & (P[:, 2] < ztop + 0.005)).any():
                    reasons.add('mesa')
                    break
            m = np.minimum(qq - e.lo, e.hi - qq)
            if not (m[3] > F.LIMIT_MARGIN and m[4] > F.LIMIT_MARGIN):
                # el arranque mismo puede estar junto al límite: solo cuenta si el camino EMPEORA el margen
                m0 = np.minimum(q - e.lo, e.hi - q)
                if min(m[3], m[4]) < min(m0[3], m0[4]) - 1e-6:
                    reasons.add('muneca_en_limite')
        ok = not reasons
        r = ','.join(sorted(reasons)) or None
        self._path_cache[key] = {'qg': qg.copy(), 'ok': ok, 'reason': r}
        return ok, r

    def committed(self, q, cands) -> float | None:
        """Afinidad de la muñeca medida (+1 sobre A, -1 sobre B) respecto de la mejor candidata de cada familia
        (la familia ausente se representa por el volteo de la otra, como en el análisis)."""
        reps = {}
        for f in BA.FAMILIES:
            fc = [v['q'] for (fam, _), v in cands.items() if fam == f]       # 'S' no representa a ninguna
            if fc:
                reps[f] = min(fc, key=lambda r: float(np.linalg.norm((r - q)[BA.WRIST])))
        if not reps:
            return None
        ra = reps.get('A', BA.mirrored_rep(reps['B']) if 'B' in reps else None)
        rb = reps.get('B', BA.mirrored_rep(reps['A']) if 'A' in reps else None)
        return BA.affinity(q, ra, rb)

    # ------------------------------------------------------------------------------------------------ decisión
    def choose(self, p):
        q = self._q()
        cands = self.candidates(q, p)
        aff = self.committed(q, cands)
        thr_a = BA.COMMIT - (FAMILY_HYST if self.family == 'A' else 0.0)      # banda de histéresis
        want = 'A' if (aff is not None and aff > thr_a) else 'B'
        decision = 'continuidad' if want == 'A' else ('preferencia' if (aff is None or aff > -BA.COMMIT)
                                                        else 'continuidad')
        order = [want, 'B' if want == 'A' else 'A']
        chosen = None
        tried = {}
        for i, fam in enumerate(order):
            keys = sorted([k for k, v in cands.items() if k[0] in (fam, 'S') and v['valid']],
                          key=lambda k: cands[k]['cost'])
            if self.family == fam and self.tilt is not None and keys:
                prev = [k for k in keys if k[1] == self.tilt]
                if prev and cands[prev[0]]['cost'] <= cands[keys[0]]['cost'] + TILT_HYST:
                    keys.remove(prev[0])
                    keys.insert(0, prev[0])
            for k in keys:
                ok, why = self.path_check(q, k, cands[k]['q'], p)
                tried[f'{k[0]}@{k[1]}'] = why
                if ok:
                    chosen = k
                    break
            if chosen is not None:
                if i == 1:
                    decision = 'respaldo_familia'
                break
        self.viability = {f'{k[0]}@{k[1]}': v['reason'] for k, v in cands.items()}
        self.viability.update({f'camino {k}': v for k, v in tried.items() if v})
        if chosen is None:
            self.decision = 'respaldo'
            self.decisions.append({'step': len(self.log), 'family': self.family, 'tilt': self.tilt,
                                   'decision': 'respaldo', 'affinity': aff})
            return None
        if self.refine:
            # refinamiento continuo de la elegida: mismo objetivo cartesiano, eje dentro de AXIS_TOL, coste =
            # distancia al estado medido + penalización suave de límites de joint4/joint5 (solo selección de IK)
            x, err = self.project_closest(cands[chosen]['q'], q, p, chosen[1], limits=True)
            # se usa SIEMPRE que sea válido (no solo cuando mejora la penalización): aplicarlo a veces sí y a veces
            # no alternaba entre dos objetivos y producía saltos de 0,15-0,2 rad
            if err < IK_TOL and self.family_key(x) in (chosen[0], 'S') and self._goal_ok(x, p):
                cands[chosen] = dict(cands[chosen], q=x, limit_pen=self.limit_penalty(x), refined=True)
        prev_goal = self.q_goal
        self.family = order[0] if decision != 'respaldo_familia' else order[1]
        self.tilt = chosen[1]
        self.decision = decision
        qg = cands[chosen]['q']
        self.decisions.append({'step': len(self.log), 'family': self.family, 'tilt': self.tilt,
                               'decision': decision, 'affinity': aff, 'cost': cands[chosen]['cost'],
                               'singular': chosen[0] == 'S',
                               'limit_pen': cands[chosen]['limit_pen'],
                               'goal_jump': None if prev_goal is None or self.decision == 'respaldo'
                               else float(np.linalg.norm(qg - prev_goal))})
        return qg

    # ------------------------------------------------------------------------------------------------ política
    def act(self) -> np.ndarray:
        e = self.env
        if e.params.goal == 'reach' and self.phase == 'pregrasp' and self.R is None:
            p = e.backend.object_pose_truth() + np.array([0.0, 0.0, e.params.pregrasp_height])
            qg = self.choose(p)
            if qg is not None:
                self.q_goal, self.p_goal = np.asarray(qg, float), p.copy()
                a = np.clip((self.q_goal - e._q_cmd) / e.delta, -1.0, 1.0)
                self.log.append({'phase': self.phase})
                return np.append(a, -1.0).astype(np.float32)
            self.p_goal = None                   # respaldo: IK global de ScriptedExpert desde aquí
        return super().act()
