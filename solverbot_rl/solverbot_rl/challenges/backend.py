"""Backends de base móvil para los desafíos HRI, GPSR y Doing Laundry.

Separa explícitamente:
  * mediciones que existen en el robot real: odometría, LiDAR, articulaciones de cabeza, cámara de cabeza;
  * verdad del simulador (`*_truth`): SOLO para recompensas, métricas, expertos de referencia y validación.

KinematicMobileBackend: SIN motor de física (unicycle 2D con límites de aceleración, colisión de la huella real contra
la geometría del mundo, LiDAR por raycasting). Sirve para pruebas unitarias y para entrenar rápido; un éxito aquí NO
es una validación física. GazeboMobileBackend (gazebo_mobile.py) es la versión física.
"""
from __future__ import annotations

import abc
import math

import numpy as np

from .arena import AXLE, LASER_XY, ArenaMap

V_MAX, W_MAX = 0.4, 1.0            # límites del diff_drive_controller (config/controllers_rl.yaml)
A_MAX, ALPHA_MAX = 0.4, 1.2        # = límites de aceleración del diff_drive_controller (controllers_rl.yaml)
PAN_LIM, TILT_LIM = 1.5708, 0.7854
LIDAR_FOV, LIDAR_MAX, LIDAR_MIN = 2.3562, 5.0, 0.06
PERSON_RADIUS = 0.22               # huella de una persona simulada (m)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def compose(p, d):
    """Pose p (x, y, yaw) compuesta con desplazamiento d expresado en el frame de p."""
    c, s = math.cos(p[2]), math.sin(p[2])
    return np.array([p[0] + c * d[0] - s * d[1], p[1] + s * d[0] + c * d[1], wrap(p[2] + d[2])])


def relative(p, q):
    """Desplazamiento de q en el frame de p."""
    c, s = math.cos(p[2]), math.sin(p[2])
    dx, dy = q[0] - p[0], q[1] - p[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(q[2] - p[2])])


class MobileBackend(abc.ABC):
    physical = False

    def __init__(self, arena: ArenaMap, lidar_beams: int = 271):
        self.arena = arena
        self.lidar_angles = np.linspace(-LIDAR_FOV, LIDAR_FOV, lidar_beams)

    # ---- control
    @abc.abstractmethod
    def reset(self, rng: np.random.Generator, pose, entities: dict) -> None:
        """pose = (x, y, yaw) de base_footprint; entities = {nombre: (x, y, yaw)} de entidades cinemáticas."""

    @abc.abstractmethod
    def cmd_vel(self, v: float, w: float) -> None: ...

    @abc.abstractmethod
    def head(self, pan: float, tilt: float) -> None: ...

    @abc.abstractmethod
    def advance(self, dt: float) -> None: ...

    # ---- mediciones del robot
    @abc.abstractmethod
    def odom(self) -> np.ndarray:
        """Pose por odometría de ruedas en el frame del mapa, inicializada con la pose conocida del reset."""

    @abc.abstractmethod
    def scan(self) -> np.ndarray:
        """Rangos del LiDAR (m), ángulos en self.lidar_angles; inf = sin retorno."""

    @abc.abstractmethod
    def head_joints(self) -> tuple[float, float]: ...

    def head_rgbd(self):
        """(rgb, depth, K, T_cam_mapa) de la cámara de cabeza, o None si el backend no tiene cámara."""
        return None

    # ---- entidades del mundo (personas, puerta de entrada, ...): las mueve el simulador del desafío
    @abc.abstractmethod
    def set_entity(self, name: str, x: float, y: float, yaw: float = 0.0, z: float = 0.0) -> None: ...

    # ---- verdad (solo recompensa / métricas / expertos)
    @abc.abstractmethod
    def pose_truth(self) -> np.ndarray: ...

    @abc.abstractmethod
    def collided_truth(self) -> bool:
        """¿Hubo contacto de la huella del robot con el mundo desde el último reset?"""

    def tipped_truth(self) -> bool:
        """¿La base volcó? (el modelo cinemático no puede volcar)."""
        return False

    def dynamic_obstacles(self) -> list:
        return []

    def close(self) -> None:
        pass


class KinematicMobileBackend(MobileBackend):
    physical = False

    def __init__(self, arena: ArenaMap, lidar_beams: int = 271, odom_noise: float = 0.02, lidar_noise: float = 0.01,
                 dt_internal: float = 0.02):
        super().__init__(arena, lidar_beams)
        self.odom_noise, self.lidar_noise, self.dt_internal = odom_noise, lidar_noise, dt_internal
        self.entities: dict[str, np.ndarray] = {}
        self.entity_radius: dict[str, float] = {}

    def reset(self, rng, pose, entities):
        self.rng = rng
        self.pose = np.array(pose, float)
        self._odom = self.pose.copy()
        self.v = self.w = 0.0
        self.v_cmd = self.w_cmd = 0.0
        self.pan = self.tilt = 0.0
        self.pan_cmd = self.tilt_cmd = 0.0
        self.entities = {k: np.array(v, float) for k, v in entities.items()}
        self._collided = False
        self.odom_bias = 1.0 + rng.normal(0, self.odom_noise)          # error de escala de rueda por episodio

    def cmd_vel(self, v, w):
        self.v_cmd, self.w_cmd = float(np.clip(v, -V_MAX, V_MAX)), float(np.clip(w, -W_MAX, W_MAX))

    def head(self, pan, tilt):
        self.pan_cmd, self.tilt_cmd = float(np.clip(pan, -PAN_LIM, PAN_LIM)), float(np.clip(tilt, -TILT_LIM, TILT_LIM))

    def set_entity(self, name, x, y, yaw=0.0, z=0.0):
        self.entities[name] = np.array([x, y, yaw], float)

    def dynamic_obstacles(self):
        return [(e[0], e[1], self.entity_radius.get(k, PERSON_RADIUS)) for k, e in self.entities.items()
                if self.entity_radius.get(k, PERSON_RADIUS) > 0]

    def advance(self, dt):
        n = max(1, int(round(dt / self.dt_internal)))
        h = dt / n
        dyn = self.dynamic_obstacles()
        for _ in range(n):
            self.v += float(np.clip(self.v_cmd - self.v, -A_MAX * h, A_MAX * h))
            self.w += float(np.clip(self.w_cmd - self.w, -ALPHA_MAX * h, ALPHA_MAX * h))
            # unicycle en el EJE DE LAS RUEDAS (centro de giro real); base_footprint lo sigue rígidamente
            c0, s0 = math.cos(self.pose[2]), math.sin(self.pose[2])
            ax = self.pose[0] + c0 * AXLE[0] - s0 * AXLE[1]
            ay = self.pose[1] + s0 * AXLE[0] + c0 * AXLE[1]
            th = self.pose[2] + 0.5 * self.w * h
            ax, ay, yaw = ax + self.v * h * math.cos(th), ay + self.v * h * math.sin(th), wrap(self.pose[2] + self.w * h)
            c1, s1 = math.cos(yaw), math.sin(yaw)
            nxt = np.array([ax - (c1 * AXLE[0] - s1 * AXLE[1]), ay - (s1 * AXLE[0] + c1 * AXLE[1]), yaw])
            if self.arena.footprint_collides(nxt[0], nxt[1], nxt[2], dynamic=dyn):
                self._collided = True          # contacto: la base se detiene (sin atravesar el obstáculo)
                self.v = self.w = 0.0
                continue
            d = relative(self.pose, nxt)
            self.pose = nxt
            self._odom = compose(self._odom, d * np.array([self.odom_bias, self.odom_bias, 1.0]))
            step = 3.0 * h                    # servos de cabeza: ~3 rad/s
            self.pan += float(np.clip(self.pan_cmd - self.pan, -step, step))
            self.tilt += float(np.clip(self.tilt_cmd - self.tilt, -step, step))

    def odom(self):
        return self._odom.copy()

    def head_joints(self):
        return self.pan, self.tilt

    def pose_truth(self):
        return self.pose.copy()

    def collided_truth(self):
        return self._collided

    def scan(self):
        return raycast_scan(self.arena, self.pose, self.lidar_angles, self.dynamic_obstacles(),
                            self.rng, self.lidar_noise)


def raycast_scan(arena: ArenaMap, pose, angles, dynamic, rng=None, noise=0.0, step=None):
    """LiDAR simulado por marcha de rayos sobre la grilla del plano del láser (+ personas como cilindros)."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    ox, oy = pose[0] + c * LASER_XY[0] - s * LASER_XY[1], pose[1] + s * LASER_XY[0] + c * LASER_XY[1]
    step = step or arena.res
    ts = np.arange(LIDAR_MIN, LIDAR_MAX, step)
    th = pose[2] + angles
    X = ox + np.cos(th)[:, None] * ts[None, :]
    Y = oy + np.sin(th)[:, None] * ts[None, :]
    ii = ((Y - arena.y0) // arena.res).astype(int)
    jj = ((X - arena.x0) // arena.res).astype(int)
    ny, nx = arena.lidar_occ.shape
    valid = (ii >= 0) & (jj >= 0) & (ii < ny) & (jj < nx)
    hit = np.zeros_like(valid)
    hit[valid] = arena.lidar_occ[ii[valid], jj[valid]]
    for px, py, r in dynamic:
        hit |= (X - px) ** 2 + (Y - py) ** 2 <= r * r
    first = np.where(hit.any(1), hit.argmax(1), -1)
    rng_ = np.where(first >= 0, ts[np.clip(first, 0, None)], np.inf)
    if rng is not None and noise > 0:
        rng_ = np.where(np.isfinite(rng_), rng_ + rng.normal(0, noise, rng_.shape), rng_)
    return rng_
