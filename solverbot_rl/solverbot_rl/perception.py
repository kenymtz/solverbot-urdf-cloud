"""Percepción mínima con la cámara RGB-D de cabeza: segmentación por color + profundidad -> posición 3D.

Usa SOLO lo que el robot real tendría: imagen, profundidad alineada, CameraInfo y la pose de la cámara por
cinemática directa (pan/tilt medidos). Es un detector de línea base para la primera fase; se reemplazará por un
detector aprendido (p. ej. YOLO) cuando haya más categorías de objetos.
"""
from __future__ import annotations

import numpy as np


def color_mask(rgb: np.ndarray, target_rgb, tol: float = 0.30) -> np.ndarray:
    """Máscara de píxeles cuyo color normalizado (cromaticidad) y brillo se parecen a `target_rgb` (0-1)."""
    img = rgb.astype(np.float32) / 255.0
    s = img.sum(axis=2, keepdims=True) + 1e-6
    chroma = img / s
    t = np.asarray(target_rgb, np.float32)
    tc = t / (t.sum() + 1e-6)
    close = np.linalg.norm(chroma - tc, axis=2) < tol
    bright = img.max(axis=2) > 0.25
    return close & bright


def backproject(mask: np.ndarray, depth: np.ndarray, k: np.ndarray, min_pixels: int = 15):
    """Centroide 3D (frame óptico) de los píxeles de la máscara con profundidad válida. None si no hay."""
    v, u = np.nonzero(mask)
    if u.size < min_pixels:
        return None
    z = depth[v, u].astype(np.float64)
    ok = np.isfinite(z) & (z > 0.05) & (z < 5.0)
    if ok.sum() < min_pixels:
        return None
    u, v, z = u[ok], v[ok], z[ok]
    zm = np.median(z)
    keep = np.abs(z - zm) < 0.05                       # descarta bordes/fondo
    u, v, z = u[keep], v[keep], z[keep]
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    pts = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)
    return pts.mean(axis=0)


def locate_object(rgb, depth, k, cam_to_base: np.ndarray, target_rgb):
    """Posición del objeto en el frame base (o None)."""
    if rgb is None or depth is None or k is None:
        return None
    if depth.shape[:2] != rgb.shape[:2]:
        return None
    p = backproject(color_mask(rgb, target_rgb), depth, k)
    if p is None:
        return None
    return (cam_to_base @ np.append(p, 1.0))[:3]


def locate_on_plane(rgb, k, cam_to_base: np.ndarray, target_rgb, plane_z: float, min_pixels: int = 15):
    """Cámara monocular (p. ej. la de muñeca): rayo por el centroide de la máscara intersectado con el plano
    horizontal z = plane_z (altura del centro del objeto apoyado en la mesa). None si no hay detección."""
    if rgb is None or k is None:
        return None
    v, u = np.nonzero(color_mask(rgb, target_rgb))
    if u.size < min_pixels:
        return None
    ray = np.array([(u.mean() - k[0, 2]) / k[0, 0], (v.mean() - k[1, 2]) / k[1, 1], 1.0])
    d = cam_to_base[:3, :3] @ ray
    o = cam_to_base[:3, 3]
    if d[2] > -1e-3 or o[2] - plane_z < 0.05:      # el rayo no baja hacia el plano o la cámara está demasiado cerca
        return None
    return o + d * (plane_z - o[2]) / d[2]
