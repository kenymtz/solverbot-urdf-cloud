#!/usr/bin/env python3
"""Mide sobre el escaneo métrico de robot_v2_description las cotas usadas en config/visual_v3.yaml.

Carga las mallas escaneadas (ya corregidas de escala, s = 1.4322) en base_link y reporta percentiles robustos
(1-99 %) de cada estructura: niveles del chasis, montantes, columna, bandeja, inclinación del monitor, cabezal y
micrófono. Con --figura guarda proyecciones frontal/lateral/planta para comparar visualmente.

Uso (fuera de ROS; requiere numpy, pyyaml, trimesh, pycollada; --figura además matplotlib):
    python3 tools/measure_v2_scan.py <raíz del repositorio> [--figura salida.png]
"""
import argparse
import os

import numpy as np
import trimesh
import yaml

PARTS = {'base': 'base_link', 'column': 'column_link', 'upper_box': 'upper_box_link', 'arm': 'arm_link',
         'camera_mount': 'camera_mount_link', 'left_wheel': 'left_wheel', 'right_wheel': 'right_wheel',
         'rear_left_caster': 'rear_left_caster', 'rear_right_caster': 'rear_right_caster'}


def load(repo):
    v2 = os.path.join(repo, 'robot_v2_description')
    frames = yaml.safe_load(open(os.path.join(v2, 'config', 'robot_geometry.yaml')))['frames']
    return {n: np.asarray(trimesh.load(os.path.join(v2, 'meshes', 'visual', f'{n}.dae'), force='mesh').vertices)
            + np.array(frames[f]) for n, f in PARTS.items()}


def pct(a, q=(1, 99)):
    return np.round(np.percentile(a, q), 3).tolist()


def report(p):
    b = p['base']
    print('CHASIS')
    for lo, hi, name in [(0.135, 0.15, 'placa inferior'), (0.165, 0.19, 'nivel inferior'), (0.29, 0.33, 'nivel superior')]:
        s = b[(b[:, 2] > lo) & (b[:, 2] < hi)]
        print(f'  {name:15s} z {lo}-{hi}: x {pct(s[:, 0], (0.5, 99.5))} y {pct(s[:, 1], (0.5, 99.5))}')
    s = b[(b[:, 2] > 0.2) & (b[:, 2] < 0.28)]
    h, xe, ye = np.histogram2d(s[:, 0], s[:, 1], bins=[np.arange(-0.28, 0.25, 0.01), np.arange(-0.26, 0.27, 0.01)])
    dense = sorted({(round(xe[i], 2), round(ye[j], 2)) for i, j in np.argwhere(h > np.percentile(h, 99.3))})
    print(f'  montantes (x, y con densidad vertical alta entre niveles): {dense}')
    c = p['column']
    m = c[(c[:, 2] > 0.5) & (c[:, 2] < 0.9)]
    print(f'COLUMNA x {pct(m[:, 0])} y {pct(m[:, 1])} z {pct(c[:, 2], (0.5, 99.5))}')
    u = p['upper_box']
    tray = u[u[:, 2] < 0.99]
    print(f'CAJA SUPERIOR x {pct(u[:, 0], (0.5, 99.5))} y {pct(u[:, 1], (0.5, 99.5))} z {pct(u[:, 2], (0.5, 99.5))}')
    print(f'  bandeja z {pct(tray[:, 2])} x {pct(tray[:, 0], (0.5, 99.5))}')
    blk = u[u[:, 2] > 1.205]
    print(f'  bloque superior x {pct(blk[:, 0], (2, 98))} y {pct(blk[:, 1], (2, 98))} z {pct(blk[:, 2], (2, 98))}')
    zs, xs = [], []
    for z0 in np.arange(1.0, 1.2, 0.025):
        s = u[(u[:, 2] > z0) & (u[:, 2] < z0 + 0.02) & (np.abs(u[:, 1]) < 0.15)]
        zs.append(z0 + 0.01)
        xs.append(np.percentile(s[:, 0], 99))
    k, x0 = np.polyfit(np.array(zs) - 1.0, xs, 1)
    print(f'  cara frontal del monitor: x = {x0:.4f} {k:+.3f} (z - 1.0)  -> inclinación {np.degrees(np.arctan(-k)):.1f}°')
    h = p['camera_mount']
    for z0, z1, name in [(1.32, 1.345, 'base pan'), (1.36, 1.40, 'cámara'), (1.42, 1.46, 'micrófono')]:
        s = h[(h[:, 2] > z0) & (h[:, 2] < z1)]
        print(f'CABEZAL {name:9s} z {z0}-{z1}: x {pct(s[:, 0], (2, 98))} y {pct(s[:, 1], (2, 98))}')
    a = p['arm']
    print(f'BRAZO (pose del escaneo) x {pct(a[:, 0])} y {pct(a[:, 1])} z {pct(a[:, 2])}')


def figure(p, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(24, 11))
    for a, (title, i, j, flip) in zip(ax, [('frente', 1, 2, True), ('lateral izquierdo', 0, 2, False), ('planta', 0, 1, False)]):
        for pts in p.values():
            s = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 30000), replace=False)]
            a.scatter(-s[:, i] if flip else s[:, i], s[:, j], s=0.2, alpha=0.4)
        a.set_aspect('equal')
        a.set_title(title)
        a.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=60)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('repo', nargs='?', default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument('--figura')
    args = ap.parse_args()
    parts = load(args.repo)
    report(parts)
    if args.figura:
        figure(parts, args.figura)
