"""Experto guionado (IK + posición REAL del objeto). Uso exclusivo: comprobar que cada nivel es resoluble y que
las métricas de éxito/recompensa lo detectan (cota superior), y servir de línea base. No es una política entrenada
ni usa la observación de la política.

Secuencia de agarre (basada en los experimentos de Gazebo/WSL del 2026-10-27):
  pregrasp  pinza en precierre (gripper_open = 32 mm, lejos del límite de 35 mm), orientación elegida por episodio
            con el eje de cierre perpendicular a dos caras del cubo; transición con error < 4 mm XY y < 6 mm Z.
  descend   pasos cartesianos de 1 cm hasta centro + grasp_dz, orientación fija, incrementos articulares escalados
            uniformemente (no recortados por eje).
  close     orden de cierre: el entorno baja el objetivo con rampa; se espera a que los dedos queden DETENIDOS por el
            objeto (medido) y se sigue apretando `squeeze_steps` pasos. Si se cierran sin objeto: reintento.
  lift      subida de 1 cm por paso (microlevantamiento) hasta +10 cm.
"""
from __future__ import annotations

import numpy as np

from .kinematics import solve_ik

APPROACH = np.array([np.sin(np.radians(20)), 0.0, -np.cos(np.radians(20))])
TILTS_DEG = (0, 10, 20, 30)
SQUEEZE_STEPS = 6
MAX_RETRIES = 1


def grasp_rotations(obj_yaw: float):
    """Orientaciones candidatas del TCP (columnas x, y = eje de cierre, z = aproximación) para un cubo con `obj_yaw`:
    eje de cierre horizontal y perpendicular a un par de caras; aproximación inclinada `tilt` alrededor de ese eje.
    Ordenadas de la más vertical a la más inclinada."""
    zw = np.array([0.0, 0.0, 1.0])
    out = []
    for tilt in TILTS_DEG:
        th = np.radians(tilt)
        for k in range(4):
            psi = obj_yaw + k * np.pi / 2
            c = np.array([np.cos(psi), np.sin(psi), 0.0])
            h = np.cross(zw, c)
            for sgn in ((1.0,) if tilt == 0 else (1.0, -1.0)):
                a = -np.cos(th) * zw + np.sin(th) * sgn * h
                x = np.cross(c, a)
                out.append((tilt, np.column_stack([x, c, a])))
    return out


def rotation_error_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    cos = (np.trace(r_a.T @ r_b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


class ScriptedExpert:
    def __init__(self, env):
        self.env = env.unwrapped
        self.reset()

    def reset(self):
        self.phase, self.count, self.q_goal, self.p_goal = 'pregrasp', 0, None, None
        self.R = None                    # orientación de agarre elegida para el episodio
        self.descent = None              # waypoints articulares verificados del descenso
        self.descent_timeout = 60        # pasos máximos de descenso
        self.retries = 0
        self.lift_start = None
        self.log: list[dict] = []        # trazas por paso (validación física)

    # ------------------------------------------------------------------------------------------------ IK
    def _fixed(self):
        return {'torso_lift_joint': self.env.cfg['robot']['lift']}

    def _plan_descent(self, q_start, pre, grasp, rot, n=8):
        """Waypoints articulares de pre -> grasp en línea recta con orientación fija; None si alguno no se alcanza
        (< 2 mm y < 3°) o si hay saltos articulares > 0.25 rad entre waypoints."""
        e = self.env
        path, q = [], q_start
        for k in range(1, n + 1):
            p = pre + (grasp - pre) * k / n
            q_new, err = solve_ik(e.chain, 'tcp_link', e.arm_joints, q, p, rot, self._fixed(), rot_weight=0.3,
                                  reg=0.005)
            r = e.chain.link_pose('tcp_link', {**self._fixed(), **dict(zip(e.arm_joints, q_new))})[:3, :3]
            if err > 0.002 or rotation_error_deg(r, rot) > 3.0 or np.max(np.abs(q_new - q)) > 0.25:
                return None
            path.append(q_new)
            q = q_new
        return path

    def _choose_grasp(self, obj, pre):
        """Primera orientación (más vertical) con pre-agarre alcanzable y descenso recto verificado."""
        e = self.env
        grasp = obj + np.array([0.0, 0.0, e.cfg.get('task', {}).get('grasp_dz', 0.0)])
        for tilt, rot in grasp_rotations(e.backend.object_yaw_truth()):
            q1, err1 = solve_ik(e.chain, 'tcp_link', e.arm_joints, e._q_cmd, pre, rot, self._fixed(), rot_weight=0.3,
                                restarts=3, reg=0.002)
            r1 = e.chain.link_pose('tcp_link', {**self._fixed(), **dict(zip(e.arm_joints, q1))})[:3, :3]
            if err1 > 0.002 or rotation_error_deg(r1, rot) > 3.0:
                continue
            path = self._plan_descent(q1, pre, grasp, rot)
            if path is not None:
                self.descent = path
                self.descent_timeout = 5 * len(path) + 20
                return rot, q1
        return None, None

    def _ik(self, p):
        """Solución global (multi-arranque) para el objetivo cartesiano de la fase; se recalcula si cambia > 5 mm."""
        if self.p_goal is not None and np.linalg.norm(p - self.p_goal) < 0.005:
            return self.q_goal
        e = self.env
        if self.R is not None:
            q, _ = solve_ik(e.chain, 'tcp_link', e.arm_joints, e._q_cmd, p, self.R, self._fixed(), rot_weight=0.3,
                            restarts=3, reg=0.002)
            self.p_goal, self.q_goal = p.copy(), q
            return q
        best = None
        for tilt in (20, 30, 40):
            ax = np.array([np.sin(np.radians(tilt)), 0.0, -np.cos(np.radians(tilt))])
            q, err = solve_ik(e.chain, 'tcp_link', e.arm_joints, e._q_cmd, p, None, self._fixed(), approach_axis=ax,
                              rot_weight=1.0, restarts=4, reg=0.002)
            if best is None or err < best[1]:
                best = (q, err)
            if err < 0.003:
                break
        self.p_goal, self.q_goal = p.copy(), best[0]
        return self.q_goal

    # ------------------------------------------------------------------------------------------------ política
    def act(self) -> np.ndarray:
        e = self.env
        s = e._state
        obj = e.backend.object_pose_truth()
        tgt = np.array([*e.params.target_xy, e.params.table_height + e.params.obj_half_height])
        goal = e.params.goal
        grasp_dz = e.cfg.get('task', {}).get('grasp_dz', 0.0)
        grip = -1.0
        if self.phase == 'pregrasp':
            p = obj + np.array([0, 0, e.params.pregrasp_height])
            if goal != 'reach' and self.R is None and hasattr(e.backend, 'object_yaw_truth'):
                self.R, q_pre = self._choose_grasp(obj, p)
                if q_pre is not None:
                    self.p_goal, self.q_goal = p.copy(), q_pre
            if (goal != 'reach'
                    and np.linalg.norm((s.tcp_pos - p)[:2]) < 0.004
                    and abs(s.tcp_pos[2] - p[2]) < 0.006):
                self.phase = 'descend'
        if self.phase == 'descend':
            p = obj + np.array([0, 0, grasp_dz])
            # tiempo máximo fijado al planificar (la lista de waypoints se consume durante el descenso)
            if np.linalg.norm(s.tcp_pos - p) < 0.006 or self.count > self.descent_timeout:
                self.phase, self.count = 'close', 0
            else:
                self.count += 1
        if self.phase == 'close':
            p, grip = s.tcp_pos.copy(), 1.0
            self.count += 1
            blocked = s.gripper_q > 0.003 and s.gripper_q - s.gripper_target > e.params.block_margin
            if blocked:
                self.phase, self.count = 'squeeze', 0
            elif s.gripper_target <= 1e-4 and s.gripper_q < 0.003:
                # se cerró sin objeto: abrir y reintentar desde el pre-agarre
                self.phase = 'reopen' if self.retries < MAX_RETRIES else 'failed'
                self.count = 0
        if self.phase == 'squeeze':
            p, grip = s.tcp_pos.copy(), 1.0
            self.count += 1
            if self.count >= SQUEEZE_STEPS:
                self.phase, self.count, self.lift_start = 'lift', 0, s.tcp_pos.copy()
        if self.phase == 'reopen':
            p, grip = s.tcp_pos + np.array([0, 0, 0.01]), -1.0
            self.count += 1
            if self.count > 25:
                self.retries += 1
                self.phase, self.count, self.p_goal = 'pregrasp', 0, None
        if self.phase == 'failed':
            p, grip = s.tcp_pos.copy(), -1.0
        if self.phase == 'lift':
            p, grip = self.lift_start + np.array([0, 0, 0.10]), 1.0
            if goal == 'place' and obj[2] - e.params.table_height > 0.07:
                self.phase = 'carry'
        if self.phase == 'carry':
            p, grip = tgt + np.array([0, 0, 0.10]), 1.0
            if np.linalg.norm(obj[:2] - tgt[:2]) < 0.015:
                self.phase = 'lower'
        if self.phase == 'lower':
            p, grip = tgt + np.array([0, 0, 0.005]), 1.0
            if obj[2] - (e.params.table_height + e.params.obj_half_height) < 0.01:
                self.phase, self.count = 'release', 0
        if self.phase == 'release':
            p, grip = s.tcp_pos.copy(), -1.0
            self.count += 1
            if self.count > 25:
                self.phase = 'retreat'
        if self.phase == 'retreat':
            p, grip = obj + np.array([0, 0, 0.12]), -1.0
        # Cerca de la mesa: pasos cartesianos de 1 cm con IK local (la interpolación articular hace arcos que
        # atraviesan la mesa). Lejos: solución global de la fase y paso articular acotado.
        if self.phase == 'descend' and self.descent is not None:
            # descenso planificado y verificado: avanzar al siguiente waypoint cuando el actual está casi alcanzado
            q_meas = np.array([e._joints()[j] for j in e.arm_joints])
            while len(self.descent) > 1 and np.max(np.abs(q_meas - self.descent[0])) < 0.01:
                self.descent.pop(0)
            q = self.descent[0]
        elif self.phase in ('descend', 'lower', 'lift', 'retreat', 'reopen') or np.linalg.norm(p - s.tcp_pos) < 0.04:
            d = p - s.tcp_pos
            n = np.linalg.norm(d)
            step = s.tcp_pos + (d if n < 0.01 else d / n * 0.01)
            if self.R is not None and goal != 'reach':
                q, _ = solve_ik(e.chain, 'tcp_link', e.arm_joints, e._q_cmd, step, self.R, self._fixed(),
                                rot_weight=0.3, reg=0.02)
            else:
                q, _ = solve_ik(e.chain, 'tcp_link', e.arm_joints, e._q_cmd, step, None, self._fixed(),
                                approach_axis=(APPROACH if goal == 'reach' else s.tcp_z), rot_weight=0.5, reg=0.02)
        else:
            q = self._ik(p)
        raw = (q - e._q_cmd) / e.delta

        # Durante el descenso y el levantamiento, conservar las proporciones del movimiento articular de la IK
        # (recortar cada eje por separado desvía la trayectoria cartesiana).
        if self.phase in ('descend', 'lift'):
            factor = max(1.0, float(np.max(np.abs(raw))))
            a = raw / factor
        else:
            a = np.clip(raw, -1.0, 1.0)

        return np.append(a, grip).astype(np.float32)
