#!/usr/bin/env python3
"""Validación estática de robot_v3_description (no necesita un grafo ROS en ejecución).

Comprueba, para todas las combinaciones de argumentos xacro:
  xacro, check_urdf (si está instalado), árbol de links/joints, nombres de frames esperados, límites,
  cinemática directa (elevador, brazo contra los parámetros MDH oficiales de piper_sdk, cabezal),
  frames de la RealSense D435, recursos (mallas), inercias, etiquetas ros2_control/Gazebo y que
  robot_v2_description no se haya modificado.

Uso:
  python3 tools/validate_v3.py                       # desde el repositorio (usa xacro del PATH o el módulo)
  python3 tools/validate_v3.py --markdown out.md     # además escribe un informe
  ros2 run robot_v3_description validate_v3.py       # sobre el paquete instalado
Código de salida 0 si todo pasa.
"""
import argparse
import itertools
import math
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------------------------- utilidades
def find_package_dir():
    """Directorio del paquete: el del repositorio (tools/..) o el share instalado."""
    cand = os.path.dirname(HERE)
    if os.path.isfile(os.path.join(cand, 'urdf', 'robot_v3.urdf.xacro')):
        return cand
    try:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory('robot_v3_description')
    except Exception:  # noqa: BLE001
        sys.exit('No se encuentra robot_v3_description')


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(axis, angle):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def homog(xyz=(0, 0, 0), rot=None):
    t = np.eye(4)
    t[:3, :3] = np.eye(3) if rot is None else rot
    t[:3, 3] = xyz
    return t


def floats(text, default='0 0 0'):
    return [float(v) for v in (text or default).split()]


# ------------------------------------------------------------------------------------------------ modelo
class Urdf:
    """Lectura mínima de un URDF y cinemática directa."""

    def __init__(self, xml_text):
        self.root = ET.fromstring(xml_text)
        self.links = {link.get('name'): link for link in self.root.findall('link')}
        self.joints = {}
        for j in self.root.findall('joint'):
            o = j.find('origin')
            a = j.find('axis')
            lim = j.find('limit')
            self.joints[j.get('name')] = {
                'type': j.get('type'),
                'parent': j.find('parent').get('link'),
                'child': j.find('child').get('link'),
                'xyz': floats(o.get('xyz') if o is not None else None),
                'rpy': floats(o.get('rpy') if o is not None else None),
                'axis': floats(a.get('xyz') if a is not None else None, '1 0 0'),
                'limit': None if lim is None else {k: float(lim.get(k)) for k in ('lower', 'upper', 'effort', 'velocity')
                                                   if lim.get(k) is not None},
                'mimic': j.find('mimic') is not None,
                'mimic_of': None if j.find('mimic') is None else (
                    j.find('mimic').get('joint'), float(j.find('mimic').get('multiplier', 1.0)),
                    float(j.find('mimic').get('offset', 0.0))),
            }
        self.child_joint = {j['child']: n for n, j in self.joints.items()}
        roots = [n for n in self.links if n not in self.child_joint]
        self.root_link = roots[0] if len(roots) == 1 else None
        self.roots = roots

    def movable(self):
        return [n for n, j in self.joints.items() if j['type'] not in ('fixed',) and not j['mimic']]

    def joint_transform(self, name, q):
        j = self.joints[name]
        t = homog(j['xyz'], rpy_matrix(*j['rpy']))
        if j['type'] in ('revolute', 'continuous'):
            t = t @ homog(rot=axis_angle(j['axis'], q))
        elif j['type'] == 'prismatic':
            t = t @ homog(np.asarray(j['axis'], float) / np.linalg.norm(j['axis']) * q)
        return t

    def value(self, jn, q):
        """Posición de jn: la de q, o la calculada con <mimic> a partir de su articulación principal."""
        m = self.joints[jn]['mimic_of']
        if m is not None and m[0] in q and jn not in q:
            return m[1] * q[m[0]] + m[2]
        return q.get(jn, 0.0)

    def pose(self, link, q=None):
        """Pose de link en el marco raíz (las articulaciones mimic siguen a su principal)."""
        q = q or {}
        t = np.eye(4)
        while link in self.child_joint:
            jn = self.child_joint[link]
            t = self.joint_transform(jn, self.value(jn, q)) @ t
            link = self.joints[jn]['parent']
        return t

    def relative(self, parent, child, q=None):
        return np.linalg.inv(self.pose(parent, q)) @ self.pose(child, q)

    def ancestors(self, link):
        out = []
        while link in self.child_joint:
            link = self.joints[self.child_joint[link]]['parent']
            out.append(link)
        return out


# ------------------------------------------------------------------------------------------------- xacro
def run_xacro(pkg, args):
    xacro_file = os.path.join(pkg, 'urdf', 'robot_v3.urdf.xacro')
    argv = [f'{k}:={v}' for k, v in args.items()]
    exe = shutil.which('xacro')
    if exe:
        res = subprocess.run([exe, xacro_file, *argv], capture_output=True, text=True)
        # Cualquier salida por stderr hace fallar a launch (Command) aunque xacro termine bien
        if res.returncode != 0 or res.stderr.strip():
            raise RuntimeError(res.stderr.strip())
        return res.stdout
    import xacro  # módulo de Python (pip install xacro)
    doc = xacro.process_file(xacro_file, mappings={k: str(v) for k, v in args.items()})
    return doc.toprettyxml(indent='  ')


# ----------------------------------------------------------------------------------- referencias externas
# Parámetros MDH oficiales del Piper estándar: piper_sdk/piper_param/constants.py, ROBOT_MDH_PRESET["piper"]
# (d, a, alpha, theta_offset). Referencia independiente del URDF para validar la cadena del brazo.
PIPER_MDH = (
    (0.123, 0.0, 0.0, 0.0),
    (0.0, 0.0, -1.5707963267948966, -3.0058060377846343),
    (0.0, 0.28502999999999995, 0.0, -1.793849405199772),
    (0.25075, -0.02198, 1.5707963267948966, 0.0),
    (0.0, 0.0, -1.5707963267948966, 0.0),
    (0.091, 0.0, 1.5707963267948966, 0.0),
)


def mdh_fk(q):
    t = np.eye(4)
    for (d, a, alpha, off), qi in zip(PIPER_MDH, q):
        ca, sa = math.cos(alpha), math.sin(alpha)
        th = qi + off
        ct, st = math.cos(th), math.sin(th)
        t = t @ np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -sa * d], [st * sa, ct * sa, ca, ca * d], [0, 0, 0, 1]])
    return t


# Frames que usa el software existente (base_athome, urg_node, Jetson) y que deben conservarse
COMPAT_FRAMES = ['base_footprint', 'base_link', 'left_wheel', 'right_wheel', 'laser', 'torso_column', 'torso_lift_link',
                 'mounting_cube_link', 'screen_link', 'pan_tilt_base_link', 'pan_link', 'tilt_link', 'camera_link',
                 'arm_base_link', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'gripper_base', 'link7', 'link8',
                 'tcp_link', 'arm_camera_link', 'arm_camera_optical_frame']
COMPAT_JOINTS = ['base_joint', 'left_wheel_joint', 'right_wheel_joint', 'torso_lift_joint', 'pan_joint', 'tilt_joint',
                 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'joint8']
D435_FRAMES = ['camera_bottom_screw_frame', 'camera_link'] + [
    f'camera_{s}_{k}' for s in ('depth', 'infra1', 'infra2', 'color') for k in ('frame', 'optical_frame')]
JETSON_JOINT_STATES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'joint8']
DEFAULT_HOME_DEG = [86.28, 173.91, -164.0, 66.98, 0.69, -146.41]   # piper_arm_controller.py (Jetson)
# Optical (REP-103): z = adelante (x del cuerpo), x = derecha (-y), y = abajo (-z)
OPTICAL_R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)


# --------------------------------------------------------------------------------------------- validador
class Report:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail=''):
        self.rows.append((name, bool(ok), detail))
        print(f"[{'OK' if ok else 'FALLA'}] {name}" + (f' — {detail}' if detail else ''))
        return ok

    @property
    def failed(self):
        return [r for r in self.rows if not r[1]]


def validate(pkg, rep, repo_root=None, v2_base='9401da9'):
    geo = yaml.safe_load(open(os.path.join(pkg, 'config', 'geometry_v3.yaml')))
    sim = yaml.safe_load(open(os.path.join(pkg, 'config', 'sim_provisional.yaml')))
    lim = yaml.safe_load(open(os.path.join(pkg, 'config', 'piper_limits.yaml')))

    # ---- 1. xacro sobre todas las combinaciones
    matrix = list(itertools.product(['none', 'mock', 'gazebo'], ['column', 'lift'], ['official', 'base_athome'],
                                    ['true', 'false']))
    models = {}
    for mode, head, limits, frames in matrix:
        args = {'ros2_control': mode, 'head_mount': head, 'arm_limits': limits, 'camera_frames': frames}
        try:
            models[(mode, head, limits, frames)] = run_xacro(pkg, args)
        except Exception as exc:  # noqa: BLE001
            rep.check(f'xacro {args}', False, str(exc)[-300:])
    rep.check(f'xacro: {len(models)}/{len(matrix)} combinaciones de argumentos generan URDF', len(models) == len(matrix))
    for bad in ({'ros2_control': 'foo'}, {'arm_limits': 'foo'}, {'head_mount': 'foo'}):
        try:
            run_xacro(pkg, bad)
            rep.check(f'xacro rechaza argumento inválido {bad}', False)
        except Exception:  # noqa: BLE001
            rep.check(f'xacro rechaza argumento inválido {bad}', True)

    base_key = ('none', 'column', 'official', 'true')
    text = models[base_key]
    u = Urdf(text)

    # ---- 2. check_urdf (urdfdom)
    check = shutil.which('check_urdf')
    if check:
        for key in [base_key, ('gazebo', 'lift', 'base_athome', 'true'), ('none', 'column', 'official', 'false')]:
            with tempfile.NamedTemporaryFile('w', suffix='.urdf', delete=False) as f:
                f.write(models[key])
            res = subprocess.run([check, f.name], capture_output=True, text=True)
            os.unlink(f.name)
            rep.check(f'check_urdf {key}', res.returncode == 0 and 'Successfully Parsed' in res.stdout,
                      res.stdout.splitlines()[0] if res.stdout else res.stderr[-200:])
    else:
        rep.check('check_urdf no instalado (liburdfdom-tools): se omite', True, 'OMITIDO')

    # ---- 3. árbol
    rep.check('una sola raíz = base_footprint', u.roots == ['base_footprint'], f'raíces: {u.roots}')
    rep.check('todos los joints conectan links existentes',
              all(j['parent'] in u.links and j['child'] in u.links for j in u.joints.values()))
    rep.check('árbol sin ciclos y conexo (links = joints + 1)', len(u.links) == len(u.joints) + 1,
              f'{len(u.links)} links, {len(u.joints)} joints')
    rep.check('cada link tiene a lo sumo un padre', len(u.child_joint) == len(u.joints))
    missing = [f for f in COMPAT_FRAMES + D435_FRAMES if f not in u.links]
    rep.check('frames de compatibilidad (base_athome, urg_node, Jetson, realsense) presentes', not missing,
              f'faltan {missing}' if missing else f'{len(COMPAT_FRAMES) + len(D435_FRAMES)} frames')
    missing_j = [j for j in COMPAT_JOINTS if j not in u.joints]
    rep.check('joints de compatibilidad presentes', not missing_j, f'faltan {missing_j}' if missing_j else '')
    rep.check('joints que publica la Jetson existen y son móviles',
              all(u.joints[j]['type'] in ('revolute', 'prismatic') for j in JETSON_JOINT_STATES))
    movable = sorted(u.movable())
    rep.check('articulaciones independientes: 12 (la pinza cuenta como una)', len(movable) == 12, ', '.join(movable))
    j8 = u.joints['joint8']
    rep.check('pinza: joint8 es mimic de joint7 con multiplicador -1 y offset 0 (relación oficial de AgileX)',
              j8['mimic_of'] == ('joint7', -1.0, 0.0) and not u.joints['joint7']['mimic'],
              str(j8['mimic_of']))
    rep.check('pinza: límites de joint8 = espejo de los de joint7 (0..0.035 m -> -0.035..0)',
              abs(j8['limit']['lower'] + u.joints['joint7']['limit']['upper']) < 1e-12
              and abs(j8['limit']['upper'] + u.joints['joint7']['limit']['lower']) < 1e-12)

    # ---- 4. límites
    lj = u.joints['torso_lift_joint']
    rep.check('torso_lift_joint prismático en +Z de la columna',
              lj['type'] == 'prismatic' and lj['axis'] == [0, 0, 1] and lj['parent'] == 'torso_column')
    rep.check('límites del elevador = valores provisionales separados (sim_provisional.yaml)',
              all(abs(lj['limit'][k] - sim['lift'][k]) < 1e-9 for k in ('lower', 'upper', 'velocity', 'effort')),
              str(lj['limit']))
    ok_lim = True
    for n, j in u.joints.items():
        if j['type'] in ('revolute', 'prismatic'):
            L = j['limit']
            ok_lim &= L is not None and L['lower'] < L['upper'] and L['effort'] > 0 and L['velocity'] > 0
            ok_lim &= abs(np.linalg.norm(j['axis']) - 1) < 1e-9
    rep.check('todas las revolutas/prismáticas: lower < upper, effort > 0, velocity > 0, eje unitario', ok_lim)
    for profile in ('official', 'base_athome'):
        um = Urdf(models[('none', 'column', profile, 'true')])
        ok = all(abs(um.joints[f'joint{i}']['limit'][k] - lim[profile][f'joint{i}'][k]) < 1e-9
                 for i in range(1, 7) for k in ('lower', 'upper', 'effort', 'velocity'))
        rep.check(f'límites del Piper = perfil "{profile}" de piper_limits.yaml', ok)
    q_home = dict(zip([f'joint{i}' for i in range(1, 7)], np.radians(DEFAULT_HOME_DEG)))
    inside = all(u.joints[n]['limit']['lower'] <= v <= u.joints[n]['limit']['upper'] for n, v in q_home.items())
    rep.check('pose DEFAULT_HOME de la Jetson dentro de los límites oficiales', inside)

    # ---- 5. cinemática: posiciones medidas (elevador y cabezal en cero)
    def world(link, q=None):
        return u.pose(link, q)[:3, 3]

    exp = {
        'left_wheel': geo['base']['wheels']['left'], 'right_wheel': geo['base']['wheels']['right'],
        'laser': np.add(geo['laser']['mount'], [0, 0, geo['laser']['focal_height']]),
        'torso_lift_link': [-0.035, -0.0026, 0.9787], 'mounting_cube_link': [0.1345, 0.0004, 1.0903],
        'arm_base_link': geo['arm_mount']['position_world'], 'pan_tilt_base_link': [0.0031, -0.0029, 1.3211],
    }
    for link, xyz in exp.items():
        err = np.linalg.norm(world(link) - np.asarray(xyz))
        rep.check(f'posición de {link} = medida/derivada', err < 1e-4, f'error {err * 1000:.3f} mm')
    # cámara: con el tilt del escaneo, el centro de la carcasa debe caer en la posición medida en v2
    q_scan = {'tilt_joint': geo['head']['tilt_at_scan']}
    housing = u.pose('camera_link', q_scan) @ np.array([0, -geo['realsense_d435']['depth_py'], 0, 1])
    err = np.linalg.norm(housing[:3] - np.array([0.0394, -0.0029, 1.3707]))
    rep.check('centro de la carcasa D435 (tilt del escaneo) = posición medida en v2', err < 1e-3, f'error {err * 1000:.2f} mm')
    R = u.pose('arm_base_link')[:3, :3]
    rep.check('base del brazo: z del Piper hacia +Y (montaje lateral izquierdo)', np.allclose(R[:, 2], [0, 1, 0], atol=1e-6))

    # ---- 6. elevador con estados simulados
    for head in ('column', 'lift'):
        uh = Urdf(models[('none', head, 'official', 'true')])
        moving = ['torso_lift_link', 'mounting_cube_link', 'screen_link', 'arm_base_link', 'link6', 'tcp_link']
        moving += ['pan_tilt_base_link', 'camera_link', 'camera_color_optical_frame'] if head == 'lift' else []
        fixed = ['base_link', 'laser', 'torso_column', 'left_wheel', 'right_wheel']
        fixed += ['pan_tilt_base_link', 'camera_link', 'camera_color_optical_frame'] if head == 'column' else []
        qa = {**q_home}
        for dz in (sim['lift']['lower'], 0.05, sim['lift']['upper']):
            qb = {**q_home, 'torso_lift_joint': dz}
            ok_m = all(np.allclose(uh.pose(link, qb)[:3, 3] - uh.pose(link, qa)[:3, 3], [0, 0, dz], atol=1e-12) and
                       np.allclose(uh.pose(link, qb)[:3, :3], uh.pose(link, qa)[:3, :3], atol=1e-12) for link in moving)
            ok_f = all(np.allclose(uh.pose(link, qb), uh.pose(link, qa), atol=1e-12) for link in fixed)
            rep.check(f'elevador ({head}) {dz:+.3f} m: suben/bajan {len(moving)} frames exactamente, {len(fixed)} quietos',
                      ok_m and ok_f)

    # ---- 7. brazo: cadena URDF vs MDH oficial (piper_sdk), en poses aleatorias dentro de límites
    rng = np.random.default_rng(0)
    names = [f'joint{i}' for i in range(1, 7)]
    worst_p, worst_r = 0.0, 0.0
    poses = [np.zeros(6), np.radians(DEFAULT_HOME_DEG)]
    poses += [np.array([rng.uniform(u.joints[n]['limit']['lower'], u.joints[n]['limit']['upper']) for n in names])
              for _ in range(200)]
    for q in poses:
        qd = dict(zip(names, q))
        t_urdf = u.relative('arm_base_link', 'link6', qd)
        t_mdh = mdh_fk(q)
        worst_p = max(worst_p, np.linalg.norm(t_urdf[:3, 3] - t_mdh[:3, 3]))
        worst_r = max(worst_r, np.linalg.norm(t_urdf[:3, :3] - t_mdh[:3, :3]))
    rep.check('brazo: posición de link6 (URDF) = MDH oficial de piper_sdk en 202 poses', worst_p < 5e-4,
              f'máx {worst_p * 1000:.3f} mm')
    rep.check('brazo: orientación de link6 (URDF) = MDH oficial de piper_sdk', worst_r < 2e-3, f'máx ‖ΔR‖ {worst_r:.2e}')
    # cada articulación mueve solo su subárbol y gira sobre su eje
    ok = True
    for n in names:
        child = u.joints[n]['child']
        q1 = {n: 0.3}
        before, after = u.pose(child), u.pose(child, q1)
        ok &= np.allclose(before[:3, 3], after[:3, 3], atol=1e-12)                 # el origen del hijo no se traslada
        ax = before[:3, :3] @ np.asarray(u.joints[n]['axis'])
        rel = after[:3, :3] @ before[:3, :3].T
        ang = math.acos(max(-1, min(1, (np.trace(rel) - 1) / 2)))
        ok &= abs(ang - 0.3) < 1e-9 and np.allclose(rel @ ax, ax, atol=1e-9)
        ok &= np.allclose(u.pose(u.joints[n]['parent'], q1), u.pose(u.joints[n]['parent']), atol=1e-12)
    rep.check('brazo: cada joint1..6 gira 0.3 rad sobre su eje sin mover a su padre', ok)
    # Pinza cerrada, a mitad y abierta comandando SOLO joint7 (joint8 sale del <mimic>)
    lim7 = u.joints['joint7']['limit']
    for a in (lim7['lower'], (lim7['lower'] + lim7['upper']) / 2, lim7['upper']):
        p7 = u.relative('gripper_base', 'link7', {'joint7': a})[:3, 3]
        p8 = u.relative('gripper_base', 'link8', {'joint7': a})[:3, 3]
        d7 = p7 - u.relative('gripper_base', 'link7')[:3, 3]
        d8 = p8 - u.relative('gripper_base', 'link8')[:3, 3]
        sep = np.linalg.norm(d7 - d8)
        # tolerancia 1 µm: los rpy oficiales están redondeados (1.5707963, 3.1415926)
        ok = (abs(sep - 2 * a) < 1e-6 and np.allclose(d7, -d8, atol=1e-6) and abs(np.linalg.norm(d7) - a) < 1e-6
              and abs(d7[2]) < 1e-6)
        rep.check(f'pinza con joint7={a * 1000:.1f} mm: ambos dedos se mueven {a * 1000:.1f} mm en sentidos opuestos, '
                  f'simétricos respecto del eje', ok, f'separación {sep * 1000:.2f} mm')
    tcp = u.relative('link6', 'tcp_link')[:3, 3]
    rep.check('tcp_link a 0.1358 m sobre el eje de link6', np.allclose(tcp, [0, 0, 0.1358]))

    # ---- 8. cabezal y RealSense
    cl = u.pose('camera_link')[:3, :3]
    rep.check('pan sobre +Z y tilt sobre +Y (positivo = cámara hacia abajo)',
              u.joints['pan_joint']['axis'] == [0, 0, 1] and u.joints['tilt_joint']['axis'] == [0, 1, 0] and
              u.pose('camera_link', {'tilt_joint': 0.3})[:3, :3][2, 0] < 0)
    ok = True
    for s in ('depth', 'infra1', 'infra2', 'color'):
        ok &= np.allclose(u.relative(f'camera_{s}_frame', f'camera_{s}_optical_frame')[:3, :3], OPTICAL_R, atol=1e-9)
        ok &= np.allclose(u.relative('camera_link', f'camera_{s}_frame')[:3, :3], np.eye(3))
    rep.check('D435: 4 frames ópticos con convención REP-103 (z adelante, x derecha, y abajo)', ok)
    offs = {s: u.relative('camera_link', f'camera_{s}_frame')[:3, 3] for s in ('depth', 'infra1', 'infra2', 'color')}
    D = geo['realsense_d435']
    rep.check('D435: extrínsecos nominales de Intel (color +15 mm, infra2 -50 mm en Y)',
              np.allclose(offs['color'], [0, D['depth_to_color'], 0]) and np.allclose(offs['infra2'], [0, D['depth_to_infra2'], 0])
              and np.allclose(offs['depth'], 0) and np.allclose(offs['infra1'], 0))
    rep.check('D435: camera_link mira hacia adelante del robot con el cabezal en cero', np.allclose(cl[:, 0], [1, 0, 0]))
    zf = Urdf(models[('none', 'column', 'official', 'false')])
    gone = [f for f in D435_FRAMES[2:] if f in zf.links]
    rep.check('camera_frames:=false: quedan camera_link y la rosca; se quitan los 8 frames del driver',
              'camera_link' in zf.links and not gone, f'presentes: {gone}' if gone else '')

    # ---- 9. recursos, inercias
    meshes = [m.get('filename') for m in u.root.iter('mesh')]
    missing = [m for m in meshes if not os.path.isfile(m.replace('package://robot_v3_description', pkg))]
    foreign = [m for m in meshes if not m.startswith('package://robot_v3_description/')]
    rep.check('todas las mallas referenciadas existen dentro del paquete', not missing and not foreign,
              f'{len(meshes)} referencias' + (f'; faltan {missing}' if missing else '') + (f'; externas {foreign}' if foreign else ''))
    mesh_dir = os.path.join(pkg, 'meshes')
    files = {os.path.relpath(os.path.join(d, f), mesh_dir) for d, _, fs in os.walk(mesh_dir) for f in fs}
    used = {m.replace('package://robot_v3_description/meshes/', '') for m in meshes}
    rep.check('sin mallas sin usar en el paquete', files == used, f'sin usar: {sorted(files - used)}' if files - used else '')
    size = sum(os.path.getsize(os.path.join(mesh_dir, f)) for f in files)
    textured = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
    textured += [f for f in files if f.endswith('.dae') and '<init_from>' in open(os.path.join(mesh_dir, f)).read()]
    rep.check('paquete liviano: mallas < 15 MB, sin texturas de imagen', size < 15e6 and not textured,
              f'{size / 1e6:.1f} MB' + (f'; con texturas: {textured}' if textured else ''))
    uses_package_uri = all(m.startswith('package://') for m in meshes)
    rep.check('rutas de mallas con package:// (portables)', uses_package_uri)
    ok = True
    bad = []
    for name, link in u.links.items():
        inert = link.find('inertial')
        if inert is None:
            continue
        m = float(inert.find('mass').get('value'))
        i = inert.find('inertia')
        I = np.array([[float(i.get('ixx')), float(i.get('ixy')), float(i.get('ixz'))],
                      [float(i.get('ixy')), float(i.get('iyy')), float(i.get('iyz'))],
                      [float(i.get('ixz')), float(i.get('iyz')), float(i.get('izz'))]])
        ev = np.linalg.eigvalsh(I)
        good = m > 0 and ev.min() > 0 and ev[0] + ev[1] >= ev[2] * (1 - 1e-6)
        if not good:
            bad.append(name)
        ok &= good
    rep.check('inercias: masa > 0, tensor definido positivo y desigualdad triangular', ok, f'mal: {bad}' if bad else '')
    no_inertia = [n for n, link in u.links.items() if link.find('inertial') is None and link.find('visual') is not None]
    rep.check('todo link con geometría tiene inercia (requisito de Gazebo)', not no_inertia, str(no_inertia))

    # ---- 9b. geometría visual de la estructura: DAE = primitivas de config/visual_v3.yaml
    vis = yaml.safe_load(open(os.path.join(pkg, 'config', 'visual_v3.yaml')))
    groups = [g for g in vis if g != 'materials']
    col = np.array(geo['torso']['column']['origin'], float)
    lift = col + np.array(geo['torso']['lift']['origin_in_column'], float)
    origins = {'chassis': np.zeros(3), 'column': col, 'carriage': lift,
               'upper_box': lift + np.array(geo['torso']['box']['center_in_lift'], float)}
    bad = []
    for g in groups:
        origin = origins.get(g, np.zeros(3)) if vis[g]['frame'] == 'world' else np.zeros(3)
        pts = []
        for part in vis[g]['parts']:
            if 'box' in part:
                lo, hi = np.array(part['box']['min'], float), np.array(part['box']['max'], float)
                c, half, rot = (lo + hi) / 2, (hi - lo) / 2, np.eye(3)
            elif 'box_c' in part:
                c, half = np.array(part['box_c']['center'], float), np.array(part['box_c']['size'], float) / 2
                rot = rpy_matrix(*part['box_c'].get('rpy', [0, 0, 0]))
            else:
                cy = part['cyl']
                c, rot = np.array(cy['center'], float), rpy_matrix(*cy.get('rpy', [0, 0, 0]))
                ang = np.linspace(0, 2 * math.pi, 64, endpoint=False)
                for z in (-cy['length'] / 2, cy['length'] / 2):
                    for a in ang:
                        pts.append(c - origin + rot @ np.array([cy['radius'] * math.cos(a), cy['radius'] * math.sin(a), z]))
                continue
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        pts.append(c - origin + rot @ (half * [sx, sy, sz]))
        exp = np.array(pts)
        dae = open(os.path.join(pkg, 'meshes', 'structure', f'{g}.dae')).read()
        arrays = [np.array(a.split(), float).reshape(-1, 3)
                  for a in __import__('re').findall(r'<float_array id="g\d+pa"[^>]*>([^<]*)</float_array>', dae)]
        got = np.vstack(arrays) if arrays else np.zeros((1, 3))
        ncolors = len({tuple(vis['materials'][p['material']]) for p in vis[g]['parts']})
        if (np.abs(got.min(0) - exp.min(0)).max() > 1e-3 or np.abs(got.max(0) - exp.max(0)).max() > 1e-3
                or dae.count('<lambert>') != ncolors):
            bad.append(g)
    referenced = {m.get('filename').split('/')[-1][:-4] for m in u.root.iter('mesh') if '/structure/' in m.get('filename')}
    rep.check(f'visual: los {len(groups)} DAE de estructura coinciden con sus primitivas (±1 mm) y colores del YAML',
              not bad, f'distintos: {bad} (regenerar con tools/make_structure_visuals.py)' if bad else '')
    rep.check('visual: todos los grupos de visual_v3.yaml se usan en el URDF', set(groups) == referenced,
              str(sorted(set(groups) ^ referenced)))
    one_visual = [n for n, link in u.links.items() if len(link.findall('visual')) > 1]
    rep.check('visual: un solo <visual> por link (RViz2 aplica el material de la primera a todas)', not one_visual,
              str(one_visual))
    # Las ruedas siguen teniendo radio y ancho calibrados en la colisión y en lo visual
    w = vis['wheel']['parts'][0]['cyl']
    rep.check('visual: neumático con radio y ancho calibrados (0.084 / 0.055)',
              abs(w['radius'] - geo['base']['wheels']['visual_radius']) < 1e-9 and abs(w['length'] - geo['base']['wheels']['width']) < 1e-9)

    # ---- 10. ros2_control y Gazebo
    for mode in ('mock', 'gazebo'):
        r = ET.fromstring(models[(mode, 'column', 'official', 'true')])
        rc = r.find('ros2_control')
        plugin = rc.find('hardware/plugin').text.strip()
        jn = sorted(j.get('name') for j in rc.findall('joint'))
        rep.check(f'ros2_control {mode}: todas las articulaciones móviles declaradas ({plugin})',
                  jn == sorted(movable + ['joint8']))
        j8rc = [j for j in rc.findall('joint') if j.get('name') == 'joint8'][0]
        params = {p.get('name'): p.text for p in j8rc.findall('param')}
        rep.check(f'ros2_control {mode}: joint8 sin interfaz de comando y declarado mimic de joint7 (-1)',
                  not j8rc.findall('command_interface') and params.get('mimic') == 'joint7'
                  and float(params.get('multiplier', 0)) == -1.0)
    ctrl = yaml.safe_load(open(os.path.join(pkg, 'config', 'ros2_controllers.yaml')))
    declared = set()
    for c in ('arm_controller', 'gripper_controller', 'lift_controller', 'head_controller'):
        declared |= set(ctrl[c]['ros__parameters']['joints'])
    rep.check('controladores de trayectoria cubren brazo, pinza, elevador y cabezal',
              declared == set(movable) - {'left_wheel_joint', 'right_wheel_joint'})
    rep.check('gripper_controller comanda solo joint7 (una orden lógica; joint8 es mimic)',
              ctrl['gripper_controller']['ros__parameters']['joints'] == ['joint7'])
    gz = models[('gazebo', 'column', 'official', 'true')]
    rep.check('Gazebo: plugin ign_ros2_control, cámara RGB-D y LiDAR presentes',
              'ign_ros2_control-system' in gz and 'rgbd_camera' in gz and 'gpu_lidar' in gz)
    rep.check('modo none: sin etiquetas ros2_control ni gazebo (descripción pura)',
              '<ros2_control' not in text and '<gazebo' not in text)

    # ---- 11. robot_v2_description sin cambios
    if repo_root and shutil.which('git'):
        res = subprocess.run(['git', '-C', repo_root, 'diff', '--stat', v2_base, '--', 'robot_v2_description'],
                             capture_output=True, text=True)
        res2 = subprocess.run(['git', '-C', repo_root, 'status', '--porcelain', '--', 'robot_v2_description'],
                              capture_output=True, text=True)
        rep.check(f'robot_v2_description idéntico al commit {v2_base} (git diff + status vacíos)',
                  res.returncode == 0 and not res.stdout.strip() and not res2.stdout.strip(),
                  (res.stdout + res2.stdout + res.stderr).strip()[:200])
    else:
        rep.check('comparación con robot_v2_description: sin repositorio git, se omite', True, 'OMITIDO')
    return u


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--package', default=None, help='directorio del paquete (por defecto: autodetectar)')
    ap.add_argument('--repo', default=None, help='raíz del repositorio git (para verificar robot_v2_description)')
    ap.add_argument('--v2-base', default='9401da9', help='commit de referencia de robot_v2_description')
    ap.add_argument('--markdown', default=None, help='escribir un informe Markdown')
    a = ap.parse_args()
    pkg = a.package or find_package_dir()
    repo = a.repo
    if repo is None and os.path.isdir(os.path.join(os.path.dirname(pkg), '.git')):
        repo = os.path.dirname(pkg)
    rep = Report()
    validate(pkg, rep, repo, a.v2_base)
    n_ok = sum(r[1] for r in rep.rows)
    print(f'\n{n_ok}/{len(rep.rows)} comprobaciones OK')
    if a.markdown:
        with open(a.markdown, 'w') as f:
            f.write('| # | Comprobación | Resultado | Detalle |\n|---|---|---|---|\n')
            for i, (name, ok, det) in enumerate(rep.rows, 1):
                f.write(f"| {i} | {name} | {'OK' if ok else '**FALLA**'} | {det.replace('|', '/')} |\n")
            f.write(f'\n**{n_ok}/{len(rep.rows)} comprobaciones OK**\n')
    sys.exit(0 if not rep.failed else 1)


if __name__ == '__main__':
    main()
