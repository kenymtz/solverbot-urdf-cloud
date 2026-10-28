"""Navegación: seguimiento de rutas del planificador y localización por LiDAR contra el mapa.

NavController (controlador de referencia, no aprendido): pure pursuit en tramos libres y, en las puertas, maniobra
de alta precisión (llegar al punto previo, girar en el lugar al eje de la puerta, avanzar recto corrigiendo el error
lateral). El vano deja ~1.5 cm por lado al robot con el Piper plegado: la maniobra no es opcional.

ScanLocalizer: estimación de pose con odometría + ajuste del LiDAR al campo de distancias del plano del láser
(Gauss-Newton). Es la pose que puede usar una política basada en sensores (no la verdad del simulador).
"""
from __future__ import annotations

import math

import numpy as np

from .arena import FOOT_CENTER, LASER_XY, ArenaMap, Waypoint
from .backend import V_MAX, W_MAX, compose, relative, wrap


def foot_center(pose) -> np.ndarray:
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.array([pose[0] + c * FOOT_CENTER[0] - s * FOOT_CENTER[1], pose[1] + s * FOOT_CENTER[0] + c * FOOT_CENTER[1]])


# Desviación máxima de rumbo al cruzar una puerta: con 1.5 cm de holgura por lado, cada grado de rumbo ocupa ~1 cm
# (largo de la huella 0.545 m); 1.5° deja corregir ~3 cm de error lateral en el cruce.
CROSS_MAX_DEV = math.radians(1.5)


class NavController:
    """Sigue una lista de Waypoint (planificados para el centro de la huella). act(pose) -> (v, w, estado)."""

    def __init__(self, waypoints: list[Waypoint], final_yaw: float | None = None, v_max: float = 0.35,
                 goal_tol: float = 0.10):
        self.wps = list(waypoints)
        self.k = 0
        self.final_yaw = final_yaw
        self.v_max = v_max
        self.goal_tol = goal_tol
        self.mode = 'track'
        self.done = False

    @property
    def in_door(self) -> bool:
        """True durante la maniobra de puerta (alineación o cruce): no se debe abandonar la ruta aquí."""
        if self.done or self.k >= len(self.wps):
            return False
        kind = self.wps[self.k].kind
        return kind == 'door_post' or (kind == 'door_pre' and self.mode == 'align')

    def act(self, pose) -> tuple[float, float]:
        if self.done or self.k >= len(self.wps):
            self.done = True
            return 0.0, 0.0
        c = foot_center(pose)
        wp = self.wps[self.k]
        d = np.array([wp.x, wp.y]) - c
        dist = float(np.linalg.norm(d))
        if wp.kind == 'door_pre':
            if self.mode == 'track':
                # el punto de referencia es el eje de las ruedas (centro de giro): girar en el lugar no lo desplaza
                if dist < 0.02:
                    self.mode = 'align'
                else:
                    return self._pursuit(pose, d, dist, slow=dist < 0.4, stop_at=True)
            if self.mode == 'align':
                err = wrap(wp.heading - pose[2])
                if abs(err) < math.radians(1.0):
                    self.mode, self.k = 'cross', self.k + 1
                    return 0.0, 0.0
                return 0.0, float(np.clip(1.5 * err, -0.5, 0.5)) if abs(err) > 0.02 else float(np.sign(err) * 0.03)
        if wp.kind == 'door_post':
            # Seguimiento de la recta del eje de la puerta (control tipo Stanley) a baja velocidad
            pre = self.wps[self.k - 1]
            n = np.array([math.cos(wp.heading), math.sin(wp.heading)])
            lat = np.array([-n[1], n[0]])
            e_lat = float((c - np.array([pre.x, pre.y])) @ lat)
            self.last_lateral_error = e_lat
            along = float((np.array([wp.x, wp.y]) - c) @ n)
            if along <= 0.0:
                self.mode, self.k = 'track', self.k + 1
                return 0.0, 0.0
            v = 0.12
            # rumbo deseado: el eje, corregido por el error lateral con una desviación máxima CROSS_MAX_DEV
            desired = wp.heading - float(np.clip(math.atan2(3.0 * e_lat, v), -CROSS_MAX_DEV, CROSS_MAX_DEV))
            w = 2.5 * wrap(desired - pose[2])
            return v, float(np.clip(w, -0.2, 0.2))
        self.mode = 'track'
        last = self.k == len(self.wps) - 1
        if last and dist < self.goal_tol:
            if self.final_yaw is not None:
                err = wrap(self.final_yaw - pose[2])
                if abs(err) > math.radians(3):
                    return 0.0, float(np.clip(1.5 * err, -0.6, 0.6))
            self.done = True
            return 0.0, 0.0
        if not last and dist < 0.15:
            self.k += 1
            return self.act(pose)
        if last and dist < 0.25:
            # aproximación final sin re-apuntar: solo avance proyectado (evita girar en torno a un punto cercano)
            ang = wrap(math.atan2(d[1], d[0]) - pose[2])
            if abs(ang) > math.radians(60):
                if dist < 0.2:
                    self.wps[self.k] = Waypoint(*foot_center(pose), 'goal')
                    return 0.0, 0.0
            else:
                return float(np.clip(0.8 * dist * math.cos(ang), 0.03, 0.15)), float(np.clip(1.5 * ang, -0.4, 0.4))
        nxt_door = self.k + 1 < len(self.wps) and self.wps[self.k + 1].kind == 'door_pre'
        return self._pursuit(pose, d, dist, slow=last and dist < 0.4, stop_at=nxt_door and dist < 0.3)

    def _pursuit(self, pose, d, dist, slow=False, stop_at=False):
        ang = wrap(math.atan2(d[1], d[0]) - pose[2])
        # girar en el lugar si el objetivo queda muy de costado, con histéresis (35° para empezar, 10° para terminar)
        turning = getattr(self, '_turning', False)
        if abs(ang) > math.radians(35) or (turning and abs(ang) > math.radians(10)):
            self._turning = True
            return 0.0, float(np.clip(1.8 * ang, -W_MAX * 0.7, W_MAX * 0.7))
        self._turning = False
        v = min(self.v_max, 0.6 * dist + 0.05) if (slow or stop_at) else self.v_max
        v *= max(0.3, math.cos(ang))
        w = float(np.clip(2.2 * ang, -W_MAX, W_MAX))
        return float(np.clip(v, 0.0, V_MAX)), w


class ScanLocalizer:
    """Localización 2D: predicción por odometría y corrección por ajuste del LiDAR al campo de distancias."""

    def __init__(self, arena: ArenaMap, angles: np.ndarray, iters: int = 6, max_range: float = 4.8,
                 subsample: int = 3, trunc: float = 0.12, prior: float = 20.0):
        """trunc: residuo (m) a partir del cual un punto se considera ajeno al mapa (personas, objetos movidos);
        prior: peso de la predicción por odometría (evita que pocos puntos arrastren la pose en zonas degeneradas)."""
        self.arena, self.angles = arena, angles
        self.iters, self.max_range, self.sub, self.trunc, self.prior = iters, max_range, subsample, trunc, prior
        self.pose = None
        self.last_odom = None
        self.last_residual = float('nan')

    def reset(self, pose, odom):
        self.pose = np.array(pose, float)
        self.last_odom = np.array(odom, float)

    def _dist(self, X, Y):
        a = self.arena
        ii = ((Y - a.y0) // a.res).astype(int).clip(0, a.lidar_dist.shape[0] - 1)
        jj = ((X - a.x0) // a.res).astype(int).clip(0, a.lidar_dist.shape[1] - 1)
        return a.lidar_dist[ii, jj]

    def update(self, odom, ranges) -> np.ndarray:
        d = relative(self.last_odom, odom)
        self.last_odom = np.array(odom, float)
        pose = compose(self.pose, d)
        r = ranges[::self.sub]
        th = self.angles[::self.sub]
        ok = np.isfinite(r) & (r < self.max_range) & (r > 0.1)
        if ok.sum() < 20:
            self.pose = pose
            return pose.copy()
        lx = LASER_XY[0] + r[ok] * np.cos(th[ok])
        ly = LASER_XY[1] + r[ok] * np.sin(th[ok])
        eps = 0.01
        pred = pose.copy()
        for _ in range(self.iters):
            c, s = math.cos(pose[2]), math.sin(pose[2])
            X, Y = pose[0] + c * lx - s * ly, pose[1] + s * lx + c * ly
            f = np.minimum(self._dist(X, Y), self.trunc)
            gx = (self._dist(X + eps, Y) - self._dist(X - eps, Y)) / (2 * eps)
            gy = (self._dist(X, Y + eps) - self._dist(X, Y - eps)) / (2 * eps)
            dXdth, dYdth = -s * lx - c * ly, c * lx - s * ly
            J = np.stack([gx, gy, gx * dXdth + gy * dYdth], 1)
            wgt = (f < self.trunc).astype(float)
            dev = pose - pred
            dev[2] = wrap(dev[2])
            P = np.diag([self.prior, self.prior, self.prior * 4])
            H = (J * wgt[:, None]).T @ J + P + np.diag([1e-3, 1e-3, 1e-3])
            step = -np.linalg.solve(H, (J * wgt[:, None]).T @ f + P @ dev)
            step = np.clip(step, -0.05, 0.05)
            pose = pose + step
            pose[2] = wrap(pose[2])
            if np.linalg.norm(step[:2]) < 1e-4 and abs(step[2]) < 1e-4:
                break
        c, s = math.cos(pose[2]), math.sin(pose[2])
        self.last_residual = float(np.mean(np.minimum(self._dist(pose[0] + c * lx - s * ly, pose[1] + s * lx + c * ly),
                                                      self.trunc)))
        self.pose = pose
        return pose.copy()
