"""Etiquetador con RAMA DE MUÑECA CONSISTENTE para `reach` (Fase B1 de exp/solverbot-wrist-branch-consistency).

OPCIONAL: solo lo usan los scripts de recolección cuando se pide (`--labeler consistent`). No reemplaza ni modifica
ScriptedExpert (solverbot_rl/expert.py), que sigue siendo el experto por defecto; tampoco cambia el entorno, la
física, las recompensas ni el criterio de éxito.

Problema (Fase A, docs/WRIST_BRANCHES.md): la IK global del experto elige entre dos familias de muñeca equivalentes
(A: joint4 > 0, B: joint4 < 0) con coste empatado y de forma no monótona en el estado inicial; la clonación por MSE
promedia (bc_v2w) o colapsa (bc_v1).

Regla de este etiquetador (función del estado articular MEDIDO y de la posición del cubo; nunca del signo
instantáneo de joint4):
  1. Familias desde soluciones de IK: para el objetivo actual se calculan las soluciones de cada familia a 20° y 30°
     (40° no cumple tcp_z[2] < -0.8, el criterio de éxito de reach).
  2. Continuidad: si la muñeca medida ya está comprometida con una familia (afinidad > COMMIT respecto de las
     soluciones de ambas familias), se conserva esa familia: nunca se fuerza una rotación completa de regreso.
  3. Si no está comprometida: la primera opción VIABLE en el orden B@20°, B@30°, A@20°, A@30° (viable = verificaciones
     físicas de branch_feasibility: IK, orientación de éxito, sin colisión con el cuerpo/propia/mesa a lo largo del
     camino del experto, muñeca lejos de sus límites, cubo visible en la cámara de muñeca al llegar).
  4. Respaldo explícito: si ninguna es viable, la IK global original de ScriptedExpert (se registra el motivo).
  5. Fase cercana (< 4 cm): IK local con el eje de aproximación de la inclinación ELEGIDA (el experto original usa
     siempre 20°, que no existe para B en parte del dominio).

Usa la verdad del simulador (pose del cubo) igual que ScriptedExpert: es un ETIQUETADOR, no una política. La política
entrenada con sus etiquetas ve solo la observación de 31 dimensiones.
"""
from __future__ import annotations

import numpy as np

from ..expert import ScriptedExpert
from ..kinematics import solve_ik
from . import branch_analysis as BA
from . import branch_feasibility as F

PREFERENCE = (('B', 20), ('B', 30), ('A', 20), ('A', 30))
COMMIT = BA.COMMIT


class BranchConsistentLabelExpert(ScriptedExpert):
    def __init__(self, env, preference=PREFERENCE, geometry: F.RobotGeometry | None = None):
        self.preference = tuple(preference)
        self.geom = geometry or F.RobotGeometry()
        super().__init__(env)

    def reset(self):
        super().reset()
        self.family = None              # familia elegida en el episodio
        self.tilt = None
        self.decision = None            # cómo se eligió: 'continuidad' | 'preferencia' | 'respaldo'
        self.viability = None           # resultado de las verificaciones por opción (caché por episodio)
        self.decisions: list[dict] = []
        self._table = None
        self._best, self._best_p = None, None    # soluciones por (familia, inclinación) del objetivo actual

    # ------------------------------------------------------------------------------------------------ familias
    def _solutions(self, q, p):
        if self._best is not None and np.linalg.norm(p - self._best_p) < 0.005:
            return self._best
        sols = BA.ik_solution_set(self.env, q, p, n_starts=6, tilts=(20, 30))
        best = {}
        for s in sols:
            k = (s['family'], s['tilt'])
            if k not in best or np.linalg.norm(s['q'] - q) < np.linalg.norm(best[k] - q):
                best[k] = s['q']
        self._best, self._best_p = best, np.asarray(p, float).copy()
        return best

    def committed_family(self, q, best: dict) -> str | None:
        """Familia con la que la muñeca MEDIDA ya está comprometida (afinidad > COMMIT respecto de la solución más
        cercana de cada familia), o None. Función de (q, objetivo): no depende del estado inicial del episodio."""
        reps = {}
        for f in BA.FAMILIES:
            cand = [v for (fam, _), v in best.items() if fam == f]
            if cand:
                reps[f] = min(cand, key=lambda r: float(np.linalg.norm((r - q)[BA.WRIST])))
        if len(reps) < 2:
            return None
        aff = BA.affinity(q, reps['A'], reps['B'])
        return 'A' if aff > COMMIT else 'B' if aff < -COMMIT else None

    def _viable(self, q, p, fam, tilt):
        e = self.env
        if self._table is None:
            self._table = F.table_in_base(e)
        cube = p - np.array([0.0, 0.0, e.params.pregrasp_height])
        r = F.check(e, self.geom, q, cube, tilt, fam, self._table)
        return r

    def choose(self, p):
        """Decide (familia, inclinación, q objetivo) desde el estado articular medido."""
        e = self.env
        q = np.array([e._joints()[j] for j in e.arm_joints])
        best = self._solutions(q, p)
        com = self.committed_family(q, best)
        if self.family is not None and com is not None and com != self.family:
            # la muñeca medida se comprometió con la otra familia (p. ej. en DAgger): continuidad, sin forzar el regreso
            self.family, self.decision = com, 'continuidad'
            self.tilt = next((t for (f, t) in best if f == com), self.tilt)
        if self.family is None:
            if com is not None:
                self.family, self.decision = com, 'continuidad'
                self.tilt = min((t for (f, t) in best if f == com), default=20)
            else:
                self.viability = {}
                for fam, tilt in self.preference:
                    r = self._viable(q, p, fam, tilt)
                    self.viability[f'{fam}@{tilt}'] = r.get('reason')
                    if r['viable']:
                        self.family, self.tilt, self.decision = fam, tilt, 'preferencia'
                        # el objetivo es EXACTAMENTE la solución verificada (no una búsqueda distinta que podría no
                        # encontrarla: eso caía al respaldo y cambiaba de familia en silencio, semilla 121128)
                        best[(fam, tilt)] = np.asarray(r['q'], float)
                        break
                if self.family is None:
                    self.decision = 'respaldo'
        self.decisions.append({'step': len(self.log), 'family': self.family, 'tilt': self.tilt,
                               'decision': self.decision, 'committed_measured': com})
        if self.decision == 'respaldo':
            return None
        q_goal = best.get((self.family, self.tilt))
        if q_goal is None:                      # esa inclinación ya no existe desde aquí: la otra de la misma familia
            alt = [(t, v) for (f, t), v in best.items() if f == self.family]
            if not alt:                         # buscarla explícitamente antes de rendirse (nunca cambiar de familia)
                for t in (self.tilt, 30 if self.tilt == 20 else 20):
                    sol = F.family_solution(self.env, q, p, t, self.family)
                    if sol is not None:
                        alt = [(t, sol)]
                        best[(self.family, t)] = sol
                        break
            if not alt:
                return None
            self.tilt, q_goal = alt[0]
        return q_goal

    def _ik(self, p):
        e = self.env
        if e.params.goal != 'reach' or self.R is not None:
            return super()._ik(p)
        if self.p_goal is not None and np.linalg.norm(p - self.p_goal) < 0.005 and self.q_goal is not None:
            # misma meta: revisar solo la continuidad (la muñeca medida pudo comprometerse con la otra familia)
            q = np.array([e._joints()[j] for j in e.arm_joints])
            com = self.committed_family(q, self._solutions(q, p))
            if com is None or com == self.family:
                return self.q_goal
        q_goal = self.choose(p)
        if q_goal is None:
            if self.decision == 'respaldo':
                return super()._ik(p)            # respaldo explícito (ninguna opción viable desde el inicio)
            # familia ya elegida pero sin solución desde aquí: conservar el objetivo anterior (nunca cambiar de familia
            # en silencio por la IK original)
            self.decisions[-1]['sin_solucion'] = True
            return self.q_goal if self.q_goal is not None else super()._ik(p)
        self.p_goal, self.q_goal = p.copy(), np.asarray(q_goal, float)
        return self.q_goal

    # ------------------------------------------------------------------------------------- fase cercana de reach
    def act(self) -> np.ndarray:
        e = self.env
        if (e.params.goal == 'reach' and self.phase == 'pregrasp' and self.tilt is not None
                and self.family is not None and self.decision != 'respaldo'):
            s = e._state
            p = e.backend.object_pose_truth() + np.array([0, 0, e.params.pregrasp_height])
            if np.linalg.norm(p - s.tcp_pos) < 0.04:
                # igual que ScriptedExpert (pasos cartesianos de 1 cm con IK local) pero con el eje de la inclinación
                # elegida, para no empujar hacia una solución de 20° que en esa familia puede no existir
                d = p - s.tcp_pos
                n = np.linalg.norm(d)
                step = s.tcp_pos + (d if n < 0.01 else d / n * 0.01)
                q, _ = solve_ik(e.chain, 'tcp_link', e.arm_joints, e._q_cmd, step, None, self._fixed(),
                                approach_axis=BA.approach_axis(self.tilt), rot_weight=0.5, reg=0.02)
                a = np.clip((q - e._q_cmd) / e.delta, -1.0, 1.0)
                self.log.append({'phase': self.phase})
                return np.append(a, -1.0).astype(np.float32)
        return super().act()
