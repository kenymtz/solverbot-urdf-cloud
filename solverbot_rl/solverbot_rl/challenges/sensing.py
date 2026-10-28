"""Percepción de personas y objetos con la cámara de cabeza (observaciones que puede usar una política).

* SimulatedHeadDetector (backend cinemático): detecta una entidad si está dentro del campo visual de la D435 según
  pan/tilt medidos, a < 5 m, con línea de vista libre de paredes; agrega ruido y pérdidas. NO usa imágenes.
* ColorHeadDetector (Gazebo): segmentación por color sobre la imagen RGB real + profundidad alineada -> posición 3D
  (perception.locate_object), pose de la cámara por cinemática directa y la pose ESTIMADA de la base (no la verdad).
La identidad de una persona es el color de su camiseta: proxy de reconocimiento (documentado), no reconocimiento facial.
"""
from __future__ import annotations

import math

import numpy as np

from ..perception import color_mask
from .backend import compose

HFOV, VFOV = 1.2043, 0.733          # D435 color (nominal)
CAM_HEIGHT = 1.374
MAX_RANGE = 5.0


def head_ray_angles(robot_pose, pan, tilt, point):
    """Ángulo (rad) entre el eje óptico de la cabeza y la dirección a `point` (x, y, z) en el mapa (aprox.: cámara en
    el eje del pan a 1.374 m). Se usa para medir contacto visual / indicar asientos."""
    yaw = robot_pose[2] + pan
    d = np.asarray(point, float) - np.array([robot_pose[0], robot_pose[1], CAM_HEIGHT])
    axis = np.array([math.cos(yaw) * math.cos(tilt), math.sin(yaw) * math.cos(tilt), -math.sin(tilt)])
    n = np.linalg.norm(d)
    return float(math.acos(np.clip(axis @ d / max(n, 1e-6), -1, 1))), float(n)


def look_at_angles(robot_pose, point):
    """pan/tilt (sin límites) que apuntan la cabeza a `point` desde `robot_pose` (pose estimada)."""
    d = np.asarray(point, float) - np.array([robot_pose[0], robot_pose[1], CAM_HEIGHT])
    pan = math.atan2(d[1], d[0]) - robot_pose[2]
    pan = (pan + math.pi) % (2 * math.pi) - math.pi
    tilt = math.atan2(-d[2], math.hypot(d[0], d[1]))
    return pan, tilt


class SimulatedHeadDetector:
    def __init__(self, arena, rng=None, noise=0.03, dropout=0.05):
        self.arena = arena
        self.rng = rng or np.random.default_rng(0)
        self.noise, self.dropout = noise, dropout

    def _line_of_sight(self, a, b):
        n = int(np.linalg.norm(np.asarray(b) - np.asarray(a)) / 0.05) + 2
        for t in np.linspace(0.05, 0.95, n):
            x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            i, j = self.arena.to_ij(x, y)
            if not (0 <= i < self.arena.nav_occ.shape[0] and 0 <= j < self.arena.nav_occ.shape[1]):
                return False
            # solo paredes / muebles altos tapan (> 1.2 m): se usa la holgura a primitivas altas
            if self.arena.tall_occ[i, j]:
                return False
        return True

    def detect(self, true_pose, est_pose, pan, tilt, targets: dict) -> dict:
        """targets: {nombre: xyz verdadero}. Devuelve {nombre: xyz estimado en el mapa} de los visibles."""
        out = {}
        for name, xyz in targets.items():
            ang_h = math.atan2(xyz[1] - true_pose[1], xyz[0] - true_pose[0]) - (true_pose[2] + pan)
            ang_h = (ang_h + math.pi) % (2 * math.pi) - math.pi
            dist = math.hypot(xyz[0] - true_pose[0], xyz[1] - true_pose[1])
            ang_v = math.atan2(CAM_HEIGHT - xyz[2], dist) - tilt
            if abs(ang_h) > HFOV / 2 or abs(ang_v) > VFOV / 2 or dist > MAX_RANGE or dist < 0.3:
                continue
            if not self._line_of_sight(true_pose[:2], xyz[:2]) or self.rng.random() < self.dropout:
                continue
            # posición medida en el frame del robot (con ruido) y llevada al mapa con la pose ESTIMADA
            c, s = math.cos(true_pose[2]), math.sin(true_pose[2])
            dx, dy = xyz[0] - true_pose[0], xyz[1] - true_pose[1]
            rel = np.array([c * dx + s * dy, -s * dx + c * dy]) + self.rng.normal(0, self.noise * (1 + dist / 3), 2)
            ce, se = math.cos(est_pose[2]), math.sin(est_pose[2])
            out[name] = np.array([est_pose[0] + ce * rel[0] - se * rel[1], est_pose[1] + se * rel[0] + ce * rel[1],
                                  xyz[2] + self.rng.normal(0, 0.03)])
        return out


def locate_gated(rgb, depth, k, cam_in_map, target_rgb, zrange=None, box=None, min_pixels=15, tol=0.30):
    """Segmentación por color + profundidad con COMPUERTA ESPACIAL: solo cuentan los píxeles cuyo punto 3D (mapa) cae
    en zrange y en box (xmin, xmax, ymin, ymax). Evita confundir objetos y camisetas de colores parecidos.
    Devuelve el centroide (mapa) del grupo principal o None."""
    if rgb is None or depth is None or k is None or depth.shape[:2] != rgb.shape[:2]:
        return None
    v, u = np.nonzero(color_mask(rgb, target_rgb, tol))
    if u.size < min_pixels:
        return None
    z = depth[v, u].astype(np.float64)
    ok = np.isfinite(z) & (z > 0.05) & (z < MAX_RANGE)
    u, v, z = u[ok], v[ok], z[ok]
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    P = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z, np.ones_like(z)], 1) @ cam_in_map.T
    keep = np.ones(len(P), bool)
    if zrange is not None:
        keep &= (P[:, 2] >= zrange[0]) & (P[:, 2] <= zrange[1])
    if box is not None:
        keep &= (P[:, 0] >= box[0]) & (P[:, 0] <= box[1]) & (P[:, 1] >= box[2]) & (P[:, 1] <= box[3])
    P = P[keep, :3]
    if len(P) < min_pixels:
        return None
    med = np.median(P, axis=0)
    main = np.linalg.norm(P[:, :2] - med[:2], axis=1) < 0.30          # grupo principal (descarta otro blob)
    if main.sum() < min_pixels:
        return None
    return P[main].mean(axis=0)


class ColorHeadDetector:
    """Gazebo: un objetivo por color (camiseta de cada persona, color de cada objeto), con compuertas espaciales:
    personas = píxeles a altura de torso; objetos = región del mueble que se inspecciona (gates)."""

    PERSON_Z = (0.55, 1.55)

    def __init__(self, colors: dict, persons=(), min_pixels: int = 25):
        self.colors, self.persons, self.min_pixels = colors, set(persons), min_pixels

    def detect(self, rgbd, est_pose, names=None, gates: dict | None = None) -> dict:
        if rgbd is None:
            return {}
        rgb, depth, k, cam_in_base = rgbd
        c, s = math.cos(est_pose[2]), math.sin(est_pose[2])
        base = np.eye(4)
        base[:2, :2] = [[c, -s], [s, c]]
        base[:2, 3] = est_pose[:2]
        cam = base @ cam_in_base
        out = {}
        gates = gates or {}
        for name in (self.colors if names is None else names):
            zr, box = gates.get(name, (self.PERSON_Z if name in self.persons else None, None))
            p = locate_gated(rgb, depth, k, cam, self.colors[name], zr, box, self.min_pixels)
            if p is not None and np.all(np.isfinite(p)):
                out[name] = p
        return out
