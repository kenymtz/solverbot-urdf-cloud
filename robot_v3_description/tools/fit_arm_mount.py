#!/usr/bin/env python3
"""Reproduce el ajuste de la posición de montaje del Piper (config/geometry_v3.yaml, arm_mount).

Coloca la cinemática OFICIAL del Piper (mallas de colisión de meshes/piper/collision) con la orientación de
montaje de base_athome (rpy = -pi/2, 0, 0) y busca la posición que mejor la superpone a la malla escaneada del
brazo de robot_v2_description (meshes/visual/arm.dae), para varias poses articulares conocidas.
Métrica: distancia de chamfer media (recortada a 10 cm) entre nubes de puntos.

Resultado obtenido (26/10/2026):
  DEFAULT_HOME (Jetson)  chamfer 12.6 mm  montaje (0.0392, 0.1818, 1.0877)   <- adoptado
  home (named_poses)     chamfer 47.1 mm
  ready (Jetson)         chamfer 63.1 mm
  ceros                  chamfer 45.8 mm
  6 GDL (posición + orientación libre, DEFAULT_HOME): 10.4 mm, desvío de orientación < 5°
  articulaciones libres: 10.3 mm, montaje x = 0.060 (de ahí la incertidumbre de ±2 cm en X)

Uso (fuera de ROS; requiere numpy, scipy, trimesh, pycollada, pyyaml):
  python3 tools/fit_arm_mount.py <raíz del repositorio>
"""
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import yaml
from scipy.optimize import minimize
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_v3 import axis_angle, homog, rpy_matrix  # noqa: E402

# Cadena oficial (agx_arm_urdf/piper/urdf/piper_description.urdf): (hijo, xyz, rpy)
CHAIN = [('link1', (0, 0, 0.123), (0, 0, 0)),
         ('link2', (0, 0, 0), (1.5707963, -0.1357866, -3.1415926)),
         ('link3', (0.28503, 0, 0), (0, 0, -1.7938494)),
         ('link4', (-0.02198, -0.25075, 0), (1.5707963, 0, 0)),
         ('link5', (0, 0, 0), (-1.5707963, 0, 0)),
         ('link6', (8.8259e-05, -0.091, 0), (1.5707963, 0, 0))]
POSES_DEG = {'DEFAULT_HOME (Jetson)': [86.28, 173.91, -164.0, 66.98, 0.69, -146.41],
             'home (named_poses)': [72.09, 176.5, -155.15, 102.62, -31.51, -99.91],
             'ready (Jetson)': [30.96, 182.68, -98.68, 37.30, -75.12, -95.86],
             'ceros': [0, 0, 0, 0, 0, 0]}
MOUNT_RPY = (-math.pi / 2, 0.0, 0.0)


def main(repo):
    v3 = os.path.join(repo, 'robot_v3_description')
    v2 = os.path.join(repo, 'robot_v2_description')
    pts = {}
    for name in ['base_link'] + [c[0] for c in CHAIN]:
        mesh = trimesh.load(os.path.join(v3, 'meshes', 'piper', 'collision', f'{name}.stl'))
        pts[name] = trimesh.sample.sample_surface(mesh, 1500, seed=1)[0]

    def cloud(q, mount):
        t = mount
        out = [(t[:3, :3] @ pts['base_link'].T).T + t[:3, 3]]
        for (child, xyz, rpy), qi in zip(CHAIN, q):
            t = t @ homog(xyz, rpy_matrix(*rpy)) @ homog(rot=axis_angle([0, 0, 1], qi))
            out.append((t[:3, :3] @ pts[child].T).T + t[:3, 3])
        return np.vstack(out)

    geo = yaml.safe_load(open(os.path.join(v2, 'config', 'robot_geometry.yaml')))
    scan = np.asarray(trimesh.load(os.path.join(v2, 'meshes', 'visual', 'arm.dae'), force='mesh').vertices)
    scan = scan + np.asarray(geo['frames']['arm_link'])
    scan = scan[np.random.default_rng(0).choice(len(scan), 6000, replace=False)]
    tree = cKDTree(scan)

    def chamfer(c):
        d1, _ = tree.query(c)
        d2, _ = cKDTree(c).query(scan)
        return 0.5 * (np.mean(np.minimum(d1, 0.1)) + np.mean(np.minimum(d2, 0.1)))

    starts = [[0.0007, 0.185, 1.0763], [0.03, 0.18, 1.02], [-0.03, 0.18, 1.1]]
    for label, deg in POSES_DEG.items():
        q = np.radians(deg)
        best = min((minimize(lambda x: chamfer(cloud(q, homog(x, rpy_matrix(*MOUNT_RPY)))), x0, method='Nelder-Mead',
                             options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 600}) for x0 in starts),
                   key=lambda r: r.fun)
        print(f'{label:24s} chamfer {best.fun * 1000:5.1f} mm  montaje {np.round(best.x, 4)}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
