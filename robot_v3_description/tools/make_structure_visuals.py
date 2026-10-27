#!/usr/bin/env python3
"""Genera meshes/structure/<grupo>.dae a partir de las primitivas de config/visual_v3.yaml.

Por qué mallas y no primitivas en el URDF: RViz2 aplica a todas las <visual> de un link el material de la primera,
así que un link con varias piezas de distintos colores (perfiles, placas, ruedas con llanta, cámara con lentes) se
ve de un solo color. Un DAE por link con un material por color se ve correctamente en RViz y en Gazebo, sin agregar
frames ni tocar la TF. Las primitivas siguen siendo la fuente (YAML); este script es determinista y el validador
comprueba que cada DAE coincide con su YAML.

Uso (fuera de ROS; requiere numpy, pyyaml, trimesh):
    python3 tools/make_structure_visuals.py [robot_v3_description]
"""
import os
import sys

import numpy as np
import trimesh
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_piper_visuals import collada  # noqa: E402
from validate_v3 import homog, rpy_matrix  # noqa: E402

CYLINDER_SECTIONS = 32


def link_origins(geo):
    """Posición en base_link (elevador y cabezal en cero) de los links con piezas en coordenadas "world"."""
    col = np.array(geo['torso']['column']['origin'], float)
    lift = col + np.array(geo['torso']['lift']['origin_in_column'], float)
    box = lift + np.array(geo['torso']['box']['center_in_lift'], float)
    return {'chassis': np.zeros(3), 'column': col, 'carriage': lift, 'upper_box': box}


def part_mesh(part, origin):
    if 'box' in part:
        lo, hi = np.array(part['box']['min'], float), np.array(part['box']['max'], float)
        return trimesh.creation.box(extents=hi - lo, transform=homog((lo + hi) / 2 - origin))
    if 'box_c' in part:
        b = part['box_c']
        t = homog(np.array(b['center'], float) - origin, rpy_matrix(*b.get('rpy', [0, 0, 0])))
        return trimesh.creation.box(extents=b['size'], transform=t)
    c = part['cyl']
    t = homog(np.array(c['center'], float) - origin, rpy_matrix(*c.get('rpy', [0, 0, 0])))
    return trimesh.creation.cylinder(radius=c['radius'], height=c['length'], sections=CYLINDER_SECTIONS, transform=t)


def group_meshes(vis, geo, group):
    """[(rgb 0-255, malla)] del grupo, en el marco del link."""
    origin = link_origins(geo).get(group, np.zeros(3)) if vis[group]['frame'] == 'world' else np.zeros(3)
    by_color = {}
    for part in vis[group]['parts']:
        rgb = tuple(int(round(255 * c)) for c in vis['materials'][part['material']][:3])
        by_color.setdefault(rgb, []).append(part_mesh(part, origin))
    out = []
    for rgb, meshes in by_color.items():
        mesh = trimesh.util.concatenate(meshes)
        mesh.unmerge_vertices()          # normales por cara: aristas nítidas en cajas
        out.append((rgb, mesh))
    return out


def main(pkg):
    vis = yaml.safe_load(open(os.path.join(pkg, 'config', 'visual_v3.yaml')))
    geo = yaml.safe_load(open(os.path.join(pkg, 'config', 'geometry_v3.yaml')))
    dst = os.path.join(pkg, 'meshes', 'structure')
    os.makedirs(dst, exist_ok=True)
    for group in [g for g in vis if g != 'materials']:
        parts = group_meshes(vis, geo, group)
        path = os.path.join(dst, f'{group}.dae')
        with open(path, 'w') as f:
            f.write(collada(parts))
        faces = sum(len(m.faces) for _, m in parts)
        print(f'{group:14s} {len(vis[group]["parts"]):2d} piezas, {len(parts)} colores, {faces:5d} caras, '
              f'{os.path.getsize(path) // 1024} KB')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
