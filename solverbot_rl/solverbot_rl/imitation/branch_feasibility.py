"""Factibilidad FÍSICA de cada familia de muñeca en `reach` (Fase B0 de exp/solverbot-wrist-branch-consistency).

SOLO ANÁLISIS: geometría y cinemática del URDF, sin simulador ni política. Para un cubo, una pose inicial, una
inclinación del eje de aproximación y una familia (branch_analysis: A joint4 > 0, B joint4 < 0), verifica:

  * solución de IK de pre-agarre dentro de los límites articulares (IK acotada) y su margen a los límites;
  * criterio de éxito de `reach` en esa pose: tcp_z[2] < -0.8 (inclinación < 36.9°; 40° nunca cumple);
  * camino que ejecuta el experto (interpolación articular con paso acotado por articulación, como ScriptedExpert)
    sin auto-colisión del brazo contra el cuerpo del robot ni contra sí mismo, sin tocar la mesa y con la muñeca
    lejos de sus límites (continuidad);
  * visibilidad del cubo en la cámara de muñeca VIRTUAL (mismo frame que gazebo_harmonic.xacro) en la pose final y
    a lo largo del camino.

Geometría de colisión: la del URDF (cajas del cuerpo; mallas STL del Piper muestreadas por vértices). Es una
aproximación conservadora por puntos con margen, no un motor de colisiones; se calibra con poses conocidas
(home, demostraciones exitosas) en el informe.
"""
from __future__ import annotations

import math
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np

from ..config import load_urdf
from ..kinematics import UrdfChain, solve_ik
from . import branch_analysis as B

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SUCCESS_TCP_Z = -0.8                     # task.success para reach (no se modifica: solo se verifica)
LIMIT_MARGIN = 0.02                      # rad: margen mínimo a los límites para considerar la muñeca "libre"
COLL_MARGIN = 0.01                       # m
BODY_LINKS = ('base_link', 'torso_column', 'mounting_cube_link', 'laser_mount', 'pan_link', 'camera_link')
ARM_LINKS = ('link3', 'link4', 'link5', 'link6', 'gripper_base', 'link7', 'link8', 'arm_camera_link')
DISTAL = ('link5', 'link6', 'gripper_base', 'link7', 'link8', 'arm_camera_link')
PROXIMAL = ('link2', 'link3')
# cámara de muñeca virtual (gazebo_harmonic.xacro + sim_cameras.yaml + geometry_v3.yaml)
_WC = (0.025, 0.0, 0.080)
_AIM = (0.0, 0.0, 0.19)
WRIST_PITCH = math.atan2(-(_AIM[2] - _WC[2]), _AIM[0] - _WC[0])
WRIST_HFOV = 1.40
WRIST_NEAR, WRIST_FAR = 0.02, 2.0


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _read_stl(path):
    with open(path, 'rb') as fh:
        data = fh.read()
    if data[:5] == b'solid' and b'facet' in data[:400]:
        v = [list(map(float, line.split()[1:4])) for line in data.decode(errors='ignore').splitlines()
             if line.strip().startswith('vertex')]
        return np.unique(np.round(np.array(v), 5), axis=0)
    n = struct.unpack('<I', data[80:84])[0]
    arr = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([('n', '<3f4'), ('v', '<9f4'), ('a', '<u2')]))
    return np.unique(np.round(arr['v'].reshape(-1, 3).astype(float), 5), axis=0)


class RobotGeometry:
    """Puntos de colisión del brazo (por link, en su frame) y cajas del cuerpo, desde el URDF del repositorio."""

    def __init__(self, urdf_xml: str | None = None, max_pts: int = 300):
        xml = urdf_xml or load_urdf()
        self.chain = UrdfChain(xml)
        root = ET.fromstring(xml)
        self.points, self.boxes = {}, {}
        for link in root.findall('link'):
            name = link.get('name')
            for c in link.findall('collision'):
                o = c.find('origin')
                xyz = np.array([float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()])
                rpy = [float(v) for v in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
                g = c.find('geometry')[0]
                if g.tag == 'box':
                    half = np.array([float(v) for v in g.get('size').split()]) / 2
                    if name in BODY_LINKS:
                        self.boxes[name] = (xyz, _rpy(*rpy), half)
                    if name in ARM_LINKS:
                        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
                        self.points[name] = (_rpy(*rpy) @ corners.T).T + xyz
                elif g.tag == 'mesh' and name in set(ARM_LINKS) | set(PROXIMAL):
                    fn = g.get('filename').replace('package://robot_v3_description', os.path.join(REPO,
                                                                                                 'robot_v3_description'))
                    p = _read_stl(fn)
                    if g.get('scale'):
                        p = p * np.array([float(v) for v in g.get('scale').split()])
                    p = p[np.linspace(0, len(p) - 1, min(max_pts, len(p))).astype(int)]
                    self.points[name] = (_rpy(*rpy) @ p.T).T + xyz

    def link_points(self, q_full: dict) -> dict:
        out = {}
        for name, P in self.points.items():
            T = self.chain.link_pose(name, q_full)
            out[name] = (T[:3, :3] @ P.T).T + T[:3, 3]
        return out

    def body_collision(self, pts: dict, margin: float = COLL_MARGIN, q_full: dict | None = None) -> list[str]:
        """Links del brazo con puntos dentro de una caja del cuerpo (+margen)."""
        hits = []
        for bname, (xyz, R, half) in self.boxes.items():
            T = self.chain.link_pose(bname, q_full or {})
            Rw, cw = T[:3, :3] @ R, T[:3, :3] @ xyz + T[:3, 3]
            for lname, P in pts.items():
                if lname not in ARM_LINKS:
                    continue
                L = (P - cw) @ Rw
                if np.any(np.all(np.abs(L) < half + margin, axis=1)):
                    hits.append(f'{lname}/{bname}')
        return hits

    def self_collision(self, pts: dict, margin: float = COLL_MARGIN) -> list[str]:
        """Links distales del brazo a menos de `margin` de los proximales (link2, link3)."""
        hits = []
        for d in DISTAL:
            if d not in pts:
                continue
            for p in PROXIMAL:
                if p not in pts:
                    continue
                dd = np.linalg.norm(pts[d][:, None, :] - pts[p][None, :, :], axis=2).min()
                if dd < margin:
                    hits.append(f'{d}/{p}')
        return hits


def wrist_camera_pose(chain: UrdfChain, q_full: dict) -> np.ndarray:
    """Frame ÓPTICO de la cámara de muñeca virtual (z hacia adelante, x derecha, y abajo), igual que el xacro."""
    A = chain.link_pose('arm_camera_link', q_full)
    T = np.eye(4)
    T[:3, 3] = (0.0, 0.0, 0.0145)
    T[:3, :3] = _rpy(0.0, WRIST_PITCH, 0.0)
    O = np.eye(4)
    O[:3, :3] = _rpy(-math.pi / 2, 0.0, -math.pi / 2)
    return A @ T @ O


def cube_in_view(cam: np.ndarray, cube: np.ndarray, hfov: float = WRIST_HFOV) -> bool:
    p = cam[:3, :3].T @ (np.asarray(cube) - cam[:3, 3])
    if not (WRIST_NEAR < p[2] < WRIST_FAR):
        return False
    t = math.tan(hfov / 2)
    return abs(p[0] / p[2]) < t and abs(p[1] / p[2]) < t        # imagen cuadrada (96 x 96): VFOV = HFOV


def expert_path(q0, q_goal, delta: float, max_steps: int = 200) -> np.ndarray:
    """Camino articular del experto hacia q_goal: cada articulación avanza a lo sumo `delta` por paso."""
    q = np.asarray(q0, float).copy()
    out = [q.copy()]
    for _ in range(max_steps):
        d = np.asarray(q_goal) - q
        if np.max(np.abs(d)) < 1e-6:
            break
        q = q + np.clip(d, -delta, delta)
        out.append(q.copy())
    return np.array(out)


def family_solution(env, q0, p, tilt: int, family: str, n_starts: int = 12, seed: int = 0):
    """Solución de pre-agarre de esa familia a esa inclinación, de menor L2 desde q0 (None si no existe)."""
    sols = [s for s in B.ik_solution_set(env, q0, p, n_starts=n_starts, seed=seed, tilts=(tilt,))
            if s['family'] == family]
    if not sols:
        return None
    return min(sols, key=lambda s: float(np.linalg.norm(s['q'] - q0)))['q']


def check(env, geom: RobotGeometry, q0, cube, tilt: int, family: str, table_base: tuple) -> dict:
    """Todas las verificaciones para (cubo, q0, inclinación, familia). cube en el frame base."""
    u = env.unwrapped
    p = np.asarray(cube, float) + np.array([0.0, 0.0, u.params.pregrasp_height])
    q = family_solution(env, q0, p, tilt, family)
    row = {'tilt': tilt, 'family': family, 'ik': q is not None}
    if q is None:
        row['viable'] = False
        row['reason'] = 'sin_solucion_ik'
        return row
    lift = u.cfg['robot']['lift']
    full = {**dict(zip(u.arm_joints, q)), 'torso_lift_joint': lift}
    tcp = geom.chain.link_pose('tcp_link', full)
    margin = np.minimum(q - u.lo, u.hi - q)
    path = expert_path(q0, q, u.delta)
    coll_body, coll_self, table_hits, view, wrist_free = set(), set(), 0, [], True
    x0, x1, y0, y1, ztop = table_base
    for qq in path:
        f = {**dict(zip(u.arm_joints, qq)), 'torso_lift_joint': lift}
        pts = geom.link_points(f)
        coll_body.update(geom.body_collision(pts, q_full=f))
        coll_self.update(geom.self_collision(pts))
        for P in pts.values():
            inside = (P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1) & (P[:, 2] < ztop + 0.005)
            table_hits += int(inside.any())
        view.append(cube_in_view(wrist_camera_pose(geom.chain, f), cube))
        m = np.minimum(qq - u.lo, u.hi - qq)
        wrist_free &= bool(m[3] > LIMIT_MARGIN and m[4] > LIMIT_MARGIN)
    row.update({
        'q': q.tolist(), 'tcp_err_mm': float(1000 * np.linalg.norm(tcp[:3, 3] - p)),
        'tcp_z2': float(tcp[2, 2]), 'success_orientation': bool(tcp[2, 2] < SUCCESS_TCP_Z),
        'margin_min': float(margin.min()), 'margin_j4': float(margin[3]), 'margin_j5': float(margin[4]),
        'path_steps': int(len(path) - 1), 'collision_body': sorted(coll_body), 'collision_self': sorted(coll_self),
        'table_contact_steps': table_hits, 'wrist_free_along_path': wrist_free,
        'cube_visible_final': bool(view[-1]), 'cube_visible_frac': float(np.mean(view)),
        'l2_from_q0': float(np.linalg.norm(q - q0))})
    reasons = []
    if row['tcp_err_mm'] > 3.0:
        reasons.append('ik_impreciso')
    if not row['success_orientation']:
        reasons.append('inclinacion_fuera_de_exito')
    if coll_body:
        reasons.append('colision_cuerpo')
    if coll_self:
        reasons.append('autocolision')
    if table_hits:
        reasons.append('mesa')
    if not wrist_free:
        reasons.append('muneca_en_limite')
    if not row['cube_visible_final']:
        reasons.append('cubo_fuera_de_vista')
    row['viable'] = not reasons
    row['reason'] = ','.join(reasons) or None
    return row


def table_in_base(env) -> tuple:
    """Rectángulo de la mesa (x0, x1, y0, y1, altura de la cara superior) en el frame base, desde la config del mundo."""
    u = env.unwrapped
    t = u.cfg['world']['table']
    cx, cy = t['xy']
    sx, sy, sz = t['size']
    corners = np.array([u.backend.world_xy_to_base([cx + a * sx / 2, cy + b * sy / 2]) for a in (-1, 1) for b in (-1, 1)])
    return (float(corners[:, 0].min()), float(corners[:, 0].max()), float(corners[:, 1].min()),
            float(corners[:, 1].max()), float(sz))
