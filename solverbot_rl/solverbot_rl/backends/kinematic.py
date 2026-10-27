"""Backend CINEMÁTICO: sin motor de física. Solo para pruebas unitarias, pruebas anti reward hacking y
depuración rápida del pipeline de RL. Un agarre aquí NO demuestra que el agarre sea físicamente posible.

Modelo: las articulaciones siguen su objetivo con un retardo de primer orden y límite de velocidad; el objeto
queda sujeto si la pinza se cierra con el objeto entre los dedos (los dedos se detienen en su media anchura);
al abrir, cae a la mesa (o al piso, fuera de la mesa). Un TCP bajo y cerca empuja el objeto (modelo burdo).
"""
from __future__ import annotations

import numpy as np

from ..kinematics import UrdfChain
from .base import Backend


class KinematicBackend(Backend):
    physical = False

    def __init__(self, chain: UrdfChain, cfg: dict):
        self.chain = chain
        self.cfg = cfg
        r, w, tk = cfg['robot'], cfg['world'], cfg['task']
        self.arm_joints = r['arm_joints']
        self.lo, self.hi = chain.limits(self.arm_joints)
        self.dt_internal = 0.02
        self.vmax = 1.0                     # rad/s (orden de magnitud de los límites del Piper)
        self.table_h = tk['table_height']
        tx, ty = w['table']['xy']
        sx, sy, _ = w['table']['size']
        self.table_rect = (tx - sx / 2, tx + sx / 2, ty - sy / 2, ty + sy / 2)
        self.obj_size = np.asarray(w['object']['size'], float)
        self.grasp_radius = tk['grasp_radius']
        self.noise = cfg['observation']['perception_noise_m']
        self.dropout = cfg['observation']['perception_dropout']
        self.rng = np.random.default_rng(0)

    # ------------------------------------------------------------------------------------------------
    def reset(self, rng, obj_xy, obj_yaw, arm_q, gripper, lift, head):
        self.rng = rng
        self.q = np.clip(np.asarray(arm_q, float), self.lo, self.hi)
        self.q_cmd = self.q.copy()
        self.g = float(gripper)
        self.g_cmd = float(gripper)
        self.fixed = {'torso_lift_joint': lift, **head}
        self.obj = np.array([obj_xy[0], obj_xy[1], self.table_h + self.obj_size[2] / 2])
        self.obj_yaw = float(obj_yaw)
        self.attached: np.ndarray | None = None
        self._last_est = self.obj.copy()

    def command(self, arm_q, gripper):
        self.q_cmd = np.clip(np.asarray(arm_q, float), self.lo, self.hi)
        self.g_cmd = float(np.clip(gripper, 0.0, 0.035))

    def _local(self):
        """Objeto en el frame de tcp_link (x: ancho del dedo, y: cierre, z: aproximación; dedos en z in [-0.076, 0])."""
        t = self.chain.link_pose('tcp_link', self.joint_positions())
        return t, t[:3, :3].T @ (self.obj - t[:3, 3])

    def advance(self, dt):
        half_w = min(self.obj_size[0], self.obj_size[1]) / 2
        half_h = self.obj_size[2] / 2
        for _ in range(max(1, int(round(dt / self.dt_internal)))):
            step = np.clip(self.q_cmd - self.q, -self.vmax * self.dt_internal, self.vmax * self.dt_internal)
            self.q = self.q + step
            t, loc = self._local()
            # entre los dedos: centrado en el ancho del dedo, dentro de la apertura y del largo de los dedos
            between = abs(loc[0]) < 0.02 and abs(loc[1]) < self.g and -0.07 < loc[2] < half_h
            g_next = self.g + np.clip(self.g_cmd - self.g, -0.025, 0.025)
            if (between or self.attached is not None) and g_next < half_w <= self.g + 1e-9:
                g_next = half_w                     # los dedos se detienen contra el objeto
            self.g = g_next
            if self.attached is None and between and self.g_cmd < half_w and abs(self.g - half_w) < 1e-6:
                self.attached = loc.copy()
            if self.attached is not None:
                if self.g_cmd > half_w + 0.002:
                    self.attached = None
                    self._fall()
                else:
                    self.obj = t[:3, :3] @ self.attached + t[:3, 3]
            elif not between and abs(loc[0]) < 0.028 + half_w and -0.076 - half_h < loc[2] < half_h \
                    and abs(loc[1]) < self.g + 0.012 + half_w:
                # un dedo toca el objeto (fuera de la zona de captura): lo empuja lateralmente (modelo burdo)
                push = np.sign(loc[1]) * (self.g + 0.012 + half_w) - loc[1]
                self.obj = self.obj + t[:3, 1] * push * np.array([1, 1, 0])
                self._fall()

    def _fall(self):
        x0, x1, y0, y1 = self.table_rect
        on_table = x0 < self.obj[0] < x1 and y0 < self.obj[1] < y1
        self.obj[2] = (self.table_h if on_table else 0.0) + self.obj_size[2] / 2

    def joint_positions(self):
        return {**self.fixed, **dict(zip(self.arm_joints, self.q)), 'joint7': self.g, 'joint8': -self.g}

    def object_pose_truth(self):
        return self.obj.copy()

    def object_yaw_truth(self):
        return self.obj_yaw

    def object_estimate(self):
        """Detector SIMULADO: verdad + ruido gaussiano + pérdidas de detección. Sustituto del pipeline de
        percepción real (backends/gazebo.py) solo en este backend sin cámaras."""
        if self.rng.random() < self.dropout:
            return self._last_est.copy(), False
        self._last_est = self.obj + self.rng.normal(0.0, self.noise, 3)
        return self._last_est.copy(), True
