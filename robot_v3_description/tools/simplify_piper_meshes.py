#!/usr/bin/env python3
"""Genera las mallas VISUALES simplificadas del Piper a partir de las mallas de colisión oficiales.

Uso (fuera de ROS; requiere trimesh, fast_simplification y rtree):
    python3 tools/simplify_piper_meshes.py robot_v3_description

Las mallas de meshes/piper/collision/ son copias sin modificar de las oficiales (ver THIRD_PARTY.md).
Las de meshes/piper/visual/ se generan con este script y se usan solo para visualización: se parte de 4000 caras
por pieza y se duplica hasta que el desvío respecto de la original sea p95 < 1.5 mm y máximo < 5 mm.
"""
import os
import sys

import fast_simplification
import numpy as np
import trimesh

TARGET_FACES = 4000
MAX_P95_M = 0.0015
MAX_DEV_M = 0.005


def main(pkg):
    src = os.path.join(pkg, 'meshes', 'piper', 'collision')
    dst = os.path.join(pkg, 'meshes', 'piper', 'visual')
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(src)):
        mesh = trimesh.load(os.path.join(src, name))
        n = len(mesh.faces)
        ref_pts, _ = trimesh.sample.sample_surface(mesh, 5000, seed=1)
        target = TARGET_FACES
        while True:
            if n > target:
                v, f = fast_simplification.simplify(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int64),
                                                    target_reduction=1.0 - target / n)
            else:
                v, f = mesh.vertices, mesh.faces
            simple = trimesh.Trimesh(v, f, process=True)
            # Desvío: distancia de 5000 puntos de la superficie original a la superficie simplificada
            _, d, _ = trimesh.proximity.closest_point(simple, ref_pts)
            if n <= target or (np.percentile(d, 95) < MAX_P95_M and d.max() < MAX_DEV_M):
                break
            target *= 2
        simple.export(os.path.join(dst, name))
        print(f'{name:16s} {n:6d} -> {len(simple.faces):5d} caras  desvío p95={np.percentile(d, 95) * 1000:.2f} mm '
              f'máx={d.max() * 1000:.2f} mm  {os.path.getsize(os.path.join(dst, name)) // 1024} KB')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
