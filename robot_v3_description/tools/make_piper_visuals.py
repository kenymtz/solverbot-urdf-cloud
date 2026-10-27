#!/usr/bin/env python3
"""Genera las mallas VISUALES del Piper (meshes/piper/visual/*.dae) a partir de las mallas visuales OFICIALES.

Fuente: agilexrobotics/agx_arm_urdf @ 680fa925, piper/meshes/dae/*.dae y agx_gripper/meshes/dae/*.dae (MIT).
Para cada pieza se agrupan las sub-mallas por color, se simplifica cada grupo (decimación cuadrática) y se
exporta un COLLADA con un material plano por color: el brazo conserva su aspecto (eslabones blancos,
articulaciones negras, detalles rojos) con ~10 veces menos caras. gripper_base no tiene DAE oficial en la
versión de la pinza estándar: se usa la malla de colisión (piper_ros) con color negro.

Comprueba que cada malla generada coincide con la de colisión oficial de la misma pieza (mismo marco):
desvío p95 de puntos de la colisión a la visual.

Uso (fuera de ROS; requiere trimesh, pycollada, fast_simplification, rtree):
    python3 tools/make_piper_visuals.py <agx_arm_urdf> [robot_v3_description]
"""
import os
import sys

import fast_simplification
import numpy as np
import trimesh

TARGET_FACES = {'link2': 12000, 'link3': 8000}   # piezas largas y visibles: más detalle
DEFAULT_TARGET = 4000
SOURCES = {
    'base_link': 'piper/meshes/dae/base_link.dae', 'link1': 'piper/meshes/dae/link1.dae',
    'link2': 'piper/meshes/dae/link2.dae', 'link3': 'piper/meshes/dae/link3.dae',
    'link4': 'piper/meshes/dae/link4.dae', 'link5': 'piper/meshes/dae/link5.dae',
    'link6': 'piper/meshes/dae/link6.dae',
    'link7': 'agx_gripper/meshes/dae/gripper_link1.dae', 'link8': 'agx_gripper/meshes/dae/gripper_link2.dae',
}
GRIPPER_BASE_RGB = (0.12, 0.12, 0.12)


def color_of(geom):
    try:
        return tuple(int(round(c)) for c in np.asarray(geom.visual.material.main_color)[:3])
    except Exception:  # noqa: BLE001
        return (190, 190, 190)


MAX_P95_M = 0.0015   # desvío máximo admitido (p95) de la superficie original a la simplificada


def simplify(mesh, target):
    """Decimación con vértices soldados; duplica el objetivo hasta que el desvío p95 sea < 1.5 mm."""
    mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, process=True)
    mesh.merge_vertices()
    pts, _ = trimesh.sample.sample_surface(mesh, 3000, seed=3)
    while len(mesh.faces) > target:
        v, f = fast_simplification.simplify(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int64),
                                            target_reduction=1.0 - target / len(mesh.faces))
        out = trimesh.Trimesh(v, f, process=True)
        if np.percentile(trimesh.proximity.closest_point(out, pts)[1], 95) < MAX_P95_M:
            return out
        target *= 2
    return mesh


def build(groups, target):
    total = sum(len(m.faces) for m in groups.values())
    return [(rgb, simplify(mesh, max(150, int(target * len(mesh.faces) / total)))) for rgb, mesh in groups.items()]


def collada(parts):
    """COLLADA 1.4 mínimo: una geometría y un material lambert (color plano) por grupo de color, en metros, Z arriba."""
    fx, geo, nodes = [], [], []
    for k, (rgb, m) in enumerate(parts):
        c = ' '.join(f'{v / 255:.4f}' for v in rgb)
        fx.append(f'<effect id="fx{k}"><profile_COMMON><technique sid="common"><lambert>'
                  f'<diffuse><color>{c} 1</color></diffuse></lambert></technique></profile_COMMON></effect>')
        pos = ' '.join(f'{v:.6g}' for v in m.vertices.ravel())
        nrm = ' '.join(f'{v:.4f}' for v in m.vertex_normals.ravel())
        idx = ' '.join(str(v) for v in np.repeat(m.faces.ravel(), 2))
        nv = len(m.vertices)
        geo.append(
            f'<geometry id="g{k}"><mesh>'
            f'<source id="g{k}p"><float_array id="g{k}pa" count="{3 * nv}">{pos}</float_array><technique_common>'
            f'<accessor source="#g{k}pa" count="{nv}" stride="3"><param name="X" type="float"/><param name="Y" type="float"/>'
            f'<param name="Z" type="float"/></accessor></technique_common></source>'
            f'<source id="g{k}n"><float_array id="g{k}na" count="{3 * nv}">{nrm}</float_array><technique_common>'
            f'<accessor source="#g{k}na" count="{nv}" stride="3"><param name="X" type="float"/><param name="Y" type="float"/>'
            f'<param name="Z" type="float"/></accessor></technique_common></source>'
            f'<vertices id="g{k}v"><input semantic="POSITION" source="#g{k}p"/></vertices>'
            f'<triangles count="{len(m.faces)}" material="mat{k}"><input semantic="VERTEX" source="#g{k}v" offset="0"/>'
            f'<input semantic="NORMAL" source="#g{k}n" offset="1"/><p>{idx}</p></triangles></mesh></geometry>')
        nodes.append(f'<node id="n{k}"><instance_geometry url="#g{k}"><bind_material><technique_common>'
                     f'<instance_material symbol="mat{k}" target="#m{k}"/></technique_common></bind_material>'
                     f'</instance_geometry></node>')
    mats = ''.join(f'<material id="m{k}"><instance_effect url="#fx{k}"/></material>' for k in range(len(parts)))
    return ('<?xml version="1.0" encoding="utf-8"?>\n<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" '
            'version="1.4.1"><asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>'
            f'<library_effects>{"".join(fx)}</library_effects><library_materials>{mats}</library_materials>'
            f'<library_geometries>{"".join(geo)}</library_geometries>'
            f'<library_visual_scenes><visual_scene id="scene">{"".join(nodes)}</visual_scene></library_visual_scenes>'
            '<scene><instance_visual_scene url="#scene"/></scene></COLLADA>\n')


def main(agx, pkg):
    dst = os.path.join(pkg, 'meshes', 'piper', 'visual')
    os.makedirs(dst, exist_ok=True)
    jobs = dict(SOURCES)
    jobs['gripper_base'] = None
    for name, rel in jobs.items():
        if rel is None:
            mesh = trimesh.load(os.path.join(pkg, 'meshes', 'piper', 'collision', 'gripper_base.stl'))
            groups = {tuple(int(255 * c) for c in GRIPPER_BASE_RGB): mesh}
        else:
            src = trimesh.load(os.path.join(agx, rel))
            meshes = src.dump() if isinstance(src, trimesh.Scene) else [src]
            groups = {}
            for m in meshes:
                groups.setdefault(color_of(m), []).append(m)
            groups = {k: trimesh.util.concatenate(v) for k, v in groups.items()}
        parts = build(groups, TARGET_FACES.get(name, DEFAULT_TARGET))
        out = os.path.join(dst, f'{name}.dae')
        with open(out, 'w') as f:
            f.write(collada(parts))
        # verificación contra la colisión oficial (mismo marco del link)
        visual = trimesh.load(out, force='mesh')
        coll = trimesh.load(os.path.join(pkg, 'meshes', 'piper', 'collision', f'{name}.stl'))
        pts, _ = trimesh.sample.sample_surface(coll, 4000, seed=1)
        _, d, _ = trimesh.proximity.closest_point(visual, pts)
        print(f'{name:13s} {sum(len(m.faces) for m in groups.values()):6d} -> {len(visual.faces):5d} caras, '
              f'{len(groups)} colores, desvío vs colisión p95 {np.percentile(d, 95) * 1000:.2f} mm, '
              f'{os.path.getsize(out) // 1024} KB')


if __name__ == '__main__':
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else here)
