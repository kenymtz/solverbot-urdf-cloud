"""Cinemática directa a partir del URDF generado por xacro (sin dependencias de ROS).

Se usa como PROPIOCEPCIÓN (pose del TCP a partir de los ángulos articulares medidos), no como ground truth
del simulador: el robot real puede calcular lo mismo con sus encoders.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np


def rpy_to_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def transform(xyz, rpy) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rpy_to_matrix(*rpy)
    t[:3, 3] = xyz
    return t


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    mimic: tuple[str, float, float] | None


class UrdfChain:
    """FK de un URDF: árbol de joints, pose de cualquier link en el frame raíz."""

    def __init__(self, urdf_xml: str):
        root = ET.fromstring(urdf_xml)
        self.joints: dict[str, Joint] = {}
        self.parent_joint: dict[str, Joint] = {}
        for j in root.findall('joint'):
            o = j.find('origin')
            xyz = [float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()]
            rpy = [float(v) for v in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
            ax = j.find('axis')
            lim = j.find('limit')
            mim = j.find('mimic')
            joint = Joint(
                name=j.get('name'), type=j.get('type'), parent=j.find('parent').get('link'),
                child=j.find('child').get('link'), origin=transform(xyz, rpy),
                axis=np.array([float(v) for v in (ax.get('xyz') if ax is not None else '1 0 0').split()]),
                lower=float(lim.get('lower', 0)) if lim is not None else 0.0,
                upper=float(lim.get('upper', 0)) if lim is not None else 0.0,
                mimic=(mim.get('joint'), float(mim.get('multiplier', 1)), float(mim.get('offset', 0)))
                if mim is not None else None)
            self.joints[joint.name] = joint
            self.parent_joint[joint.child] = joint

    @classmethod
    def from_file(cls, path: str) -> 'UrdfChain':
        with open(path) as f:
            return cls(f.read())

    def limits(self, names) -> tuple[np.ndarray, np.ndarray]:
        return (np.array([self.joints[n].lower for n in names]), np.array([self.joints[n].upper for n in names]))

    def link_pose(self, link: str, q: dict[str, float]) -> np.ndarray:
        """Matriz 4x4 de `link` en el frame raíz del árbol."""
        chain = []
        while link in self.parent_joint:
            j = self.parent_joint[link]
            chain.append(j)
            link = j.parent
        t = np.eye(4)
        for j in reversed(chain):
            t = t @ j.origin
            if j.type in ('revolute', 'continuous', 'prismatic'):
                v = q.get(j.mimic[0], 0.0) * j.mimic[1] + j.mimic[2] if j.mimic else q.get(j.name, 0.0)
                m = np.eye(4)
                if j.type == 'prismatic':
                    m[:3, 3] = j.axis / np.linalg.norm(j.axis) * v
                else:
                    m[:3, :3] = axis_angle(j.axis, v)
                t = t @ m
        return t


def solve_ik(chain: UrdfChain, link: str, joints: list[str], q0: np.ndarray, target_pos: np.ndarray,
             target_rot: np.ndarray | None = None, fixed: dict[str, float] | None = None, iters: int = 200,
             rot_weight: float = 0.3, approach_axis: np.ndarray | None = None, reg: float = 0.01,
             restarts: int = 0, seed: int = 0) -> tuple[np.ndarray, float]:
    """IK por mínimos cuadrados acotados (scipy, trust-region reflective). Devuelve (q, error de posición).

    Solo para scripts de validación/demostración (agarre guionado), no la usa la política.
    target_rot: orientación completa; approach_axis: solo la dirección del eje z del link (yaw libre).
    reg: regularización hacia q0 (continuidad de trayectorias). restarts: arranques aleatorios adicionales.
    """
    from scipy.optimize import least_squares

    lo, hi = chain.limits(joints)
    q0 = np.clip(np.asarray(q0, float), lo, hi)
    fixed = dict(fixed or {})

    def residual(qv):
        t = chain.link_pose(link, {**fixed, **dict(zip(joints, qv))})
        r = [target_pos - t[:3, 3]]
        if target_rot is not None:
            r.append(rot_weight * (t[:3, :3] - target_rot).ravel())
        if approach_axis is not None:
            r.append(rot_weight * (t[:3, 2] - approach_axis))
        r.append(reg * (qv - q0))
        return np.concatenate(r)

    rng = np.random.default_rng(seed)
    best = None
    for k in range(restarts + 1):
        start = q0 if k == 0 else rng.uniform(lo, hi)
        res = least_squares(residual, np.clip(start, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), max_nfev=iters)
        if best is None or res.cost < best.cost:
            best = res
    q = best.x
    return q, float(np.linalg.norm(residual(q)[:3]))
