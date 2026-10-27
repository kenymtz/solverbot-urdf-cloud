"""Experto guionado (IK + posición REAL del objeto). Uso exclusivo: comprobar que cada nivel es resoluble y que
las métricas de éxito/recompensa lo detectan (cota superior), y servir de línea base. No es una política entrenada
ni usa la observación de la política.
"""
from __future__ import annotations

import numpy as np

from .kinematics import solve_ik

APPROACH = np.array([np.sin(np.radians(20)), 0.0, -np.cos(np.radians(20))])


class ScriptedExpert:
    def __init__(self, env):
        self.env = env.unwrapped
        self.reset()

    def reset(self):
        self.phase, self.count, self.q_goal, self.p_goal = 'pregrasp', 0, None, None

    def _ik(self, p):
        """Solución global (multi-arranque) para el objetivo cartesiano de la fase; se recalcula si cambia > 5 mm."""
        if self.p_goal is not None and np.linalg.norm(p - self.p_goal) < 0.005:
            return self.q_goal
        e = self.env
        best = None
        for tilt in (20, 30, 40):
            ax = np.array([np.sin(np.radians(tilt)), 0.0, -np.cos(np.radians(tilt))])
            q, err = solve_ik(e.chain, 'tcp_link', e.arm_joints, e._q_cmd, p, None,
                              {'torso_lift_joint': e.cfg['robot']['lift']}, approach_axis=ax, rot_weight=1.0,
                              restarts=4, reg=0.002)
            if best is None or err < best[1]:
                best = (q, err)
            if err < 0.003:
                break
        self.p_goal, self.q_goal = p.copy(), best[0]
        return self.q_goal

    def act(self) -> np.ndarray:
        e = self.env
        s = e._state
        obj = e.backend.object_pose_truth()
        tgt = np.array([*e.params.target_xy, e.params.table_height + e.params.obj_half_height])
        goal = e.params.goal
        grip = -1.0
        if self.phase == 'pregrasp':
            p = obj + np.array([0, 0, e.params.pregrasp_height])
            if np.linalg.norm(s.tcp_pos - p) < 0.015 and goal != 'reach':
                self.phase = 'descend'
        if self.phase == 'descend':
            p = obj.copy()
            if np.linalg.norm(s.tcp_pos - p) < 0.012:
                self.phase = 'close'
        if self.phase == 'close':
            p, grip = s.tcp_pos.copy(), 1.0
            self.count += 1
            if self.count > 4:
                self.phase, self.count = 'lift', 0
        if self.phase == 'lift':
            p, grip = obj + np.array([0, 0, 0.10]), 1.0
            if goal == 'place' and obj[2] - e.params.table_height > 0.07:
                self.phase = 'carry'
        if self.phase == 'carry':
            p, grip = tgt + np.array([0, 0, 0.10]), 1.0
            if np.linalg.norm(obj[:2] - tgt[:2]) < 0.015:
                self.phase = 'lower'
        if self.phase == 'lower':
            p, grip = tgt + np.array([0, 0, 0.005]), 1.0
            if obj[2] - (e.params.table_height + e.params.obj_half_height) < 0.01:
                self.phase = 'release'
        if self.phase == 'release':
            p, grip = s.tcp_pos.copy(), -1.0
            self.count += 1
            if self.count > 3:
                self.phase = 'retreat'
        if self.phase == 'retreat':
            p, grip = obj + np.array([0, 0, 0.12]), -1.0
        # Cerca de la mesa: pasos cartesianos de 1 cm con IK local (la interpolación articular hace arcos que
        # atraviesan la mesa). Lejos: solución global de la fase y paso articular acotado.
        if self.phase in ('descend', 'lower', 'lift', 'retreat') or np.linalg.norm(p - s.tcp_pos) < 0.04:
            d = p - s.tcp_pos
            n = np.linalg.norm(d)
            q, _ = solve_ik(e.chain, 'tcp_link', e.arm_joints, e._q_cmd, s.tcp_pos + (d if n < 0.01 else d / n * 0.01),
                            None, {'torso_lift_joint': e.cfg['robot']['lift']}, approach_axis=(APPROACH if goal == 'reach' else s.tcp_z),
                            rot_weight=0.5, reg=0.02)
        else:
            q = self._ik(p)
        a = np.clip((q - e._q_cmd) / e.delta, -1, 1)
        return np.append(a, grip).astype(np.float32)
