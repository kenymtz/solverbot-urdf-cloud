"""Sonda de DIAGNÓSTICO de las dos cámaras (cabeza RGB-D y muñeca RGB) sobre un GazeboBackend en marcha.

Mide, para cada paso, lo que cada cámara ve del objeto y por qué no lo ve (fuera del campo, tapado, sin
profundidad). Usa la pose REAL del objeto solo como referencia de medición: nada de lo que calcula esta sonda entra en
la observación de la política (la sonda no toca el entorno; solo se suscribe a tópicos del mismo nodo).
"""
from __future__ import annotations

import threading
import time

import numpy as np

from .perception import color_mask, locate_object, locate_on_plane

HEAD_FRAME = 'camera_color_optical_frame'
WRIST_FRAME = 'wrist_sim_camera_optical_frame'
TOPICS = {'head': '/camera/camera/color/image_raw', 'depth': '/camera/camera/aligned_depth_to_color/image_raw',
          'wrist': '/wrist_camera/image_raw'}
INFO_TOPICS = {'head': '/camera/camera/color/camera_info', 'wrist': '/wrist_camera/camera_info'}
MIN_PIXELS = 15                                  # el mismo umbral que perception.backproject / locate_on_plane


def cube_corners(center, yaw, size):
    h = np.asarray(size, float) / 2
    c, s = np.cos(yaw), np.sin(yaw)
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pts = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
    return np.asarray(center, float) + pts @ r.T


def project(points_base, cam_to_base, k):
    """Puntos del frame base -> (u, v, z) en la cámara óptica (z adelante). z <= 0: detrás de la cámara."""
    p = np.linalg.inv(cam_to_base) @ np.c_[points_base, np.ones(len(points_base))].T
    z = p[2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = k[0, 0] * p[0] / z + k[0, 2]
        v = k[1, 1] * p[1] / z + k[1, 2]
    return np.stack([u, v, z], axis=1)


def visibility(rgb, depth, k, cam_to_base, truth, yaw, size, target_rgb, est=None):
    """Clasifica la visibilidad del objeto en una imagen con la pose REAL (diagnóstico).

    status: behind | out_of_fov | occluded (algo más cerca que el objeto sobre su proyección) | too_small |
            visible (máscara >= MIN_PIXELS) ; 'detected' indica si el detector devolvió una estimación."""
    h, w = rgb.shape[:2]
    c = project(np.asarray(truth, float)[None], cam_to_base, k)[0]
    corners = project(cube_corners(truth, yaw, size), cam_to_base, k)
    in_img = (corners[:, 2] > 0) & (corners[:, 0] >= 0) & (corners[:, 0] < w) & (corners[:, 1] >= 0) & \
        (corners[:, 1] < h)
    mask = color_mask(rgb, target_rgb)
    n_mask = int(mask.sum())
    out = {'u': float(c[0]), 'v': float(c[1]), 'range_expected': float(c[2]), 'corners_in_image': int(in_img.sum()),
           'mask_pixels': n_mask, 'mask_to_projection_px': None, 'depth_at_projection': None,
           'detected': est is not None}
    if n_mask:
        vv, uu = np.nonzero(mask)
        out['mask_to_projection_px'] = float(np.hypot(uu.mean() - c[0], vv.mean() - c[1]))
    if c[2] <= 0:
        out['status'] = 'behind'
        return out
    if not (0 <= c[0] < w and 0 <= c[1] < h) and not in_img.any():
        out['status'] = 'out_of_fov'
        return out
    if depth is not None and 0 <= c[0] < w and 0 <= c[1] < h:
        u0, v0 = int(c[0]), int(c[1])
        patch = depth[max(0, v0 - 1):v0 + 2, max(0, u0 - 1):u0 + 2].astype(float)
        d = float(np.median(patch))      # -inf: más cerca que el plano near (p. ej. la placa de la columna)
        out['depth_at_projection'] = d
        # profundidad óptica (z) de la cara visible ~ z del centro - media arista
        if not np.isfinite(d) and d < 0 or (np.isfinite(d) and d < c[2] - 0.6 * max(size)):
            out['status'] = 'occluded' if n_mask < MIN_PIXELS else 'visible'
            return out
    out['status'] = 'visible' if n_mask >= MIN_PIXELS else 'too_small'
    return out


class CameraProbe:
    """Suscripciones adicionales sobre el nodo del backend: últimas imágenes, CameraInfo y registro de sellos."""

    def __init__(self, backend, keys=('head', 'depth', 'wrist')):
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CameraInfo, Image

        from .backends.gazebo import _image_to_np, _stamp
        self.be = backend
        self._to_np, self._stamp = _image_to_np, _stamp
        self.lock = threading.Lock()
        self.img: dict[str, tuple[float, np.ndarray]] = {}
        self.info: dict[str, object] = {}
        self.log: dict[str, list[tuple[float, float]]] = {k: [] for k in keys}   # (sello simulado, reloj de pared)
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        n = backend.node
        self._subs = [n.create_subscription(Image, TOPICS[k], lambda m, k=k: self._on_img(k, m), qos) for k in keys]
        self._subs += [n.create_subscription(CameraInfo, t, lambda m, k=k: self.info.__setitem__(k, m), qos)
                       for k, t in INFO_TOPICS.items()]
        # Historial articular para evaluar la FK en el instante de cada imagen (diagnóstico de latencia)
        from sensor_msgs.msg import JointState
        self.js_hist: list[tuple[float, dict]] = []
        self._subs.append(n.create_subscription(JointState, '/joint_states', self._on_js, 50))

    def _on_js(self, m):
        with self.lock:
            prev = self.js_hist[-1][1] if self.js_hist else {}
            self.js_hist.append((self._stamp(m), {**prev, **dict(zip(m.name, m.position))}))
            del self.js_hist[:-400]

    def js_at(self, t):
        """Estado articular interpolado en el instante t (None si el historial no lo cubre)."""
        with self.lock:
            h = list(self.js_hist)
        ts = [a for a, _ in h]
        i = int(np.searchsorted(ts, t))
        if i == 0 or i >= len(h):
            return None
        (t0, a), (t1, b) = h[i - 1], h[i]
        w = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        return {k: (1 - w) * a[k] + w * b.get(k, a[k]) for k in a}

    def _on_img(self, key, m):
        t = self._stamp(m)
        with self.lock:
            self.img[key] = (t, self._to_np(m))
            self.log[key].append((t, time.monotonic()))

    def k(self, cam):
        m = self.info.get(cam)
        return None if m is None else np.array(m.k, float).reshape(3, 3)

    def wait_ready(self, timeout=60.0):
        t0 = time.monotonic()
        while not (set(self.log) <= set(self.img) and set(INFO_TOPICS) <= set(self.info)):
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(f'sonda: faltan {set(self.log) - set(self.img)} / {set(INFO_TOPICS) - set(self.info)}')
            time.sleep(0.01)

    def wait_fresh(self, max_age=0.2, timeout=2.0):
        """Espera (acotado) un cuadro de cada flujo no más viejo que max_age respecto del reloj simulado."""
        t_goal = self.be._clock - max_age
        t0 = time.monotonic()
        while any(self.img[k][0] < t_goal for k in self.img) and time.monotonic() - t0 < timeout:
            time.sleep(0.002)
        return {k: float(self.be._clock - self.img[k][0]) for k in self.img}

    def measure(self, js=None, truth=None, yaw=None):
        """Mediciones independientes de ambas cámaras en el instante actual."""
        be = self.be
        js = be.joint_positions() if js is None else js
        truth = be.object_pose_truth() if truth is None else truth
        yaw = be.object_yaw_truth() if yaw is None else yaw
        size = be.cfg['world']['object']['size']
        with self.lock:
            img = dict(self.img)
        out = {'clock': float(be._clock), 'truth': np.asarray(truth, float).tolist(),
               'pan': float(js.get('pan_joint', np.nan)), 'tilt': float(js.get('tilt_joint', np.nan))}
        # cabeza: RGB-D, profundidad alineada al color
        if 'head' in img:
            rgb = img['head'][1]
            depth = img['depth'][1] if 'depth' in img else None
            cam = be.chain.link_pose(HEAD_FRAME, js)
            kh = self.k('head')
            est = locate_object(rgb, depth, kh, cam, be.obj_rgb)
            vis = visibility(rgb, depth, kh, cam, truth, yaw, size, be.obj_rgb, est)
            d = depth.astype(float) if depth is not None else np.zeros((1, 1))
            vis.update({
                'age': float(be._clock - img['head'][0]),
                'depth_age': float(be._clock - img['depth'][0]) if 'depth' in img else None,
                'rgb_depth_dt': float(img['head'][0] - img['depth'][0]) if 'depth' in img else None,
                'frac_depth_neginf': float(np.mean(np.isneginf(d))),          # más cerca que near (0.105 m)
                'frac_depth_lt_0_30': float(np.mean(np.isfinite(d) & (d < 0.30))),
                'cam_xyz': cam[:3, 3].tolist(), 'cam_axis': cam[:3, 2].tolist(),
                'est': None if est is None else np.asarray(est).tolist(),
                'err': None if est is None else float(np.linalg.norm(np.asarray(est) - truth))})
            out['head'] = vis
        # muñeca: la MISMA función que usa la observación (backend.object_estimate), sin modificar su memoria
        if 'wrist' in img:
            last = be._last_est.copy()
            est, valid = be.object_estimate()
            be._last_est = last
            cam = be.chain.link_pose(WRIST_FRAME, js)
            kw = self.k('wrist')
            vis = visibility(img['wrist'][1], None, kw, cam, truth, yaw, size, be.obj_rgb, est if valid else None)
            vis.update({'age': float(be._clock - img['wrist'][0]), 'cam_xyz': cam[:3, 3].tolist(),
                        'cam_axis': cam[:3, 2].tolist(), 'gripper_gate': bool(js.get('joint7', 0.0) >= 0.025),
                        'est': np.asarray(est).tolist() if valid else None,
                        'err': float(np.linalg.norm(np.asarray(est) - truth)) if valid else None})
            # misma proyección sobre el plano, con la pose de la cámara en el instante de la IMAGEN
            js_img = self.js_at(img['wrist'][0])
            vis['est_at_image_time'] = vis['err_at_image_time'] = None
            if js_img is not None and valid:
                cam_i = be.chain.link_pose(WRIST_FRAME, js_img)
                p = locate_on_plane(img['wrist'][1], kw, cam_i, be.obj_rgb, be.table_h + be.obj_half_h)
                if p is not None:
                    vis['est_at_image_time'] = p.tolist()
                    vis['err_at_image_time'] = float(np.linalg.norm(p - truth))
            out['wrist'] = vis
        return out

    def rates(self):
        """Frecuencia por flujo en tiempo simulado (cuadros únicos / intervalo) y en tiempo de pared."""
        res = {}
        with self.lock:
            for k, v in self.log.items():
                if len(v) < 3:
                    res[k] = None
                    continue
                st = np.array(sorted({a for a, _ in v}))
                wl = np.array([b for _, b in v])
                res[k] = {'frames': len(v), 'unique_stamps': int(st.size),
                          'sim_hz': float((st.size - 1) / max(st[-1] - st[0], 1e-9)),
                          'sim_period_median_s': float(np.median(np.diff(st))),
                          'wall_hz': float((len(wl) - 1) / max(wl[-1] - wl[0], 1e-9))}
        return res


def look_at(chain, js, point):
    """(pan, tilt) que apuntan el eje óptico de la cabeza a `point` (frame base), recortados a los límites del URDF."""
    pan_o = chain.link_pose('pan_link', js)[:3, 3]
    d = np.asarray(point, float) - pan_o
    pan = float(np.arctan2(d[1], d[0]))
    js2 = dict(js, pan_joint=pan, tilt_joint=0.0)
    tilt_o = chain.link_pose('tilt_link', js2)[:3, 3]
    d = np.asarray(point, float) - tilt_o
    tilt = float(np.arctan2(-d[2], np.hypot(d[0], d[1])))
    (plo, tlo), (phi, thi) = chain.limits(['pan_joint', 'tilt_joint'])
    return float(np.clip(pan, plo, phi)), float(np.clip(tilt, tlo, thi))
