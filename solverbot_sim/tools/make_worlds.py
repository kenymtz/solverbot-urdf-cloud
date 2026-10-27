#!/usr/bin/env python3
"""Genera los mundos SDF (Gazebo Harmonic) a partir de config/arena_spec.yaml y config/pick_place.yaml.

    python3 tools/make_worlds.py            # escribe worlds/home_arena.sdf y worlds/pick_place_training.sdf

Los mundos generados se versionan; regenerarlos tras editar los YAML. Todo es geometría primitiva (cajas,
cilindros, esferas) con colisión = visual, para que la física sea liviana y reproducible.
"""
import math
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)

WOOD = (0.55, 0.36, 0.20)
DARK = (0.18, 0.16, 0.15)
WHITE = (0.92, 0.92, 0.92)
STEEL = (0.65, 0.66, 0.68)
FABRIC = (0.75, 0.75, 0.78)


def f(v):
    return ' '.join(f'{x:.4f}' for x in v)


def material(rgb, a=1.0):
    c = f(rgb) + f' {a}'
    return f'<material><ambient>{c}</ambient><diffuse>{c}</diffuse><specular>0.1 0.1 0.1 1</specular></material>'


def geom(shape, size):
    if shape == 'box':
        return f'<box><size>{f(size)}</size></box>'
    if shape == 'cylinder':
        return f'<cylinder><radius>{size[0] / 2:.4f}</radius><length>{size[2]:.4f}</length></cylinder>'
    if shape == 'sphere':
        return f'<sphere><radius>{size[0] / 2:.4f}</radius></sphere>'
    raise ValueError(shape)


class Part:
    """Una primitiva (visual + colisión opcional) dentro de un link."""

    def __init__(self, name, shape, size, xyz, rgb, collide=True, rpy=(0, 0, 0), mu=None):
        self.name, self.shape, self.size, self.xyz, self.rgb, self.collide, self.rpy, self.mu = (
            name, shape, size, xyz, rgb, collide, rpy, mu)

    def sdf(self):
        pose = f'<pose>{f(self.xyz)} {f(self.rpy)}</pose>'
        g = geom(self.shape, self.size)
        out = f'<visual name="{self.name}_v">{pose}<geometry>{g}</geometry>{material(self.rgb)}</visual>'
        if self.collide:
            surf = ''
            if self.mu is not None:
                surf = f'<surface><friction><ode><mu>{self.mu}</mu><mu2>{self.mu}</mu2></ode></friction></surface>'
            out += f'<collision name="{self.name}_c">{pose}<geometry>{g}</geometry>{surf}</collision>'
        return out


def static_model(name, xy, yaw, parts, z=0.0):
    body = ''.join(p.sdf() for p in parts)
    return (f'<model name="{name}"><static>true</static><pose>{xy[0]:.4f} {xy[1]:.4f} {z:.4f} 0 0 {yaw:.4f}</pose>'
            f'<link name="link">{body}</link></model>\n')


def inertia(shape, size, m):
    if shape == 'box':
        x, y, z = size
        return m * (y * y + z * z) / 12, m * (x * x + z * z) / 12, m * (x * x + y * y) / 12
    if shape == 'cylinder':
        r, h = size[0] / 2, size[2]
        return m * (3 * r * r + h * h) / 12, m * (3 * r * r + h * h) / 12, m * r * r / 2
    r = size[0] / 2
    return (2 * m * r * r / 5,) * 3


def dynamic_object(name, shape, size, mass, rgb, pose, mu=1.0, publish_pose=False):
    ixx, iyy, izz = inertia(shape, size, mass)
    g = geom(shape, size)
    # PosePublisher: pose del modelo con nombre en /model/<name>/pose (solo para recompensas/métricas)
    pp = ('<plugin filename="gz-sim-pose-publisher-system" name="gz::sim::systems::PosePublisher">'
          '<publish_link_pose>false</publish_link_pose><publish_model_pose>true</publish_model_pose>'
          '<use_pose_vector_msg>true</use_pose_vector_msg><update_frequency>100</update_frequency></plugin>'
          if publish_pose else '')
    return (f'<model name="{name}"><pose>{f(pose)} 0 0 0</pose>{pp}<link name="link">'
            f'<inertial><mass>{mass}</mass><inertia><ixx>{ixx:.3e}</ixx><iyy>{iyy:.3e}</iyy><izz>{izz:.3e}</izz>'
            f'<ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>'
            f'<collision name="c"><geometry>{g}</geometry><surface><friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode>'
            f'</friction><contact><ode><kp>1e6</kp><kd>100</kd><min_depth>0.001</min_depth></ode></contact></surface>'
            f'</collision><visual name="v"><geometry>{g}</geometry>{material(rgb)}</visual></link></model>\n')


# ---------------------------------------------------------------- mobiliario (formas compuestas reconocibles)
def furniture_parts(kind, s):
    """Partes en el frame del mueble (origen en el piso, centro de la planta). s = [x, y, z]."""
    x, y, z = s
    if kind == 'table':
        t = 0.03
        legs = [Part(f'leg{i}', 'box', [0.04, 0.04, z - t], [sx * (x / 2 - 0.04), sy * (y / 2 - 0.04), (z - t) / 2], WOOD)
                for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)])]
        return [Part('top', 'box', [x, y, t], [0, 0, z - t / 2], WOOD, mu=0.8)] + legs
    if kind == 'round_table':
        t = 0.03
        return [Part('top', 'cylinder', [x, y, t], [0, 0, z - t / 2], WOOD, mu=0.8),
                Part('column', 'cylinder', [0.10, 0.10, z - t], [0, 0, (z - t) / 2], DARK),
                Part('foot', 'cylinder', [0.50, 0.50, 0.03], [0, 0, 0.015], DARK)]
    if kind in ('washer', 'dishwasher'):
        p = [Part('body', 'box', s, [0, 0, z / 2], WHITE, mu=0.8),
             Part('panel', 'box', [x * 0.9, 0.01, 0.08], [0, -y / 2 - 0.005, z - 0.06], STEEL, collide=False)]
        if kind == 'washer':
            p.append(Part('door', 'cylinder', [0.36, 0.36, 0.02], [0, -y / 2 - 0.01, z * 0.45], (0.3, 0.35, 0.4),
                          collide=False, rpy=(math.pi / 2, 0, 0)))
        else:
            p.append(Part('handle', 'box', [x * 0.6, 0.03, 0.02], [0, -y / 2 - 0.015, z - 0.15], STEEL, collide=False))
        return p
    if kind in ('shelf', 'cabinet_open'):
        n = 3
        p = [Part('back', 'box', [x, 0.02, z], [0, y / 2 - 0.01, z / 2], WOOD) if x > y else
             Part('back', 'box', [0.02, y, z], [-x / 2 + 0.01, 0, z / 2], WOOD)]
        for side in (-1, 1):
            if x > y:
                p.append(Part(f'side{side}', 'box', [0.02, y, z], [side * (x / 2 - 0.01), 0, z / 2], WOOD))
            else:
                p.append(Part(f'side{side}', 'box', [x, 0.02, z], [0, side * (y / 2 - 0.01), z / 2], WOOD))
        for i in range(n + 1):
            h = 0.02 + i * (z - 0.04) / n
            p.append(Part(f'board{i}', 'box', [x, y, 0.02], [0, 0, h], WOOD, mu=0.8))
        return p
    if kind == 'bin':
        return [Part('body', 'cylinder', s, [0, 0, z / 2], (0.35, 0.35, 0.38))]
    if kind == 'bed':
        return [Part('frame', 'box', [x, y, 0.30], [0, 0, 0.15], WOOD),
                Part('mattress', 'box', [x - 0.04, y - 0.04, z - 0.30], [0, 0, 0.30 + (z - 0.30) / 2], (0.95, 0.95, 0.97), mu=0.9),
                Part('pillow', 'box', [0.30, y * 0.7, 0.10], [-x / 2 + 0.2, 0, z + 0.05], WHITE, collide=False),
                Part('headboard', 'box', [0.05, y, 0.9], [-x / 2 - 0.025, 0, 0.45], WOOD)]
    if kind == 'cabinet_low':
        return [Part('body', 'box', s, [0, 0, z / 2], DARK, mu=0.8),
                Part('drawer', 'box', [0.01, y * 0.8, 0.12], [x / 2 + 0.005, 0, z * 0.7], STEEL, collide=False)]
    if kind == 'coat_rack':
        return [Part('pole', 'cylinder', [0.04, 0.04, z], [0, 0, z / 2], DARK),
                Part('foot', 'cylinder', [x, y, 0.03], [0, 0, 0.015], DARK),
                Part('hook', 'box', [0.30, 0.03, 0.03], [0, 0, z - 0.1], DARK, collide=False)]
    if kind == 'tv_stand':
        return [Part('body', 'box', s, [0, 0, z / 2], (0.45, 0.30, 0.18), mu=0.8),
                Part('tv', 'box', [x * 0.9, 0.05, 0.50], [0, y / 2 - 0.05, z + 0.30], (0.05, 0.05, 0.05))]
    if kind == 'sofa':
        seat = 0.45
        return [Part('seat', 'box', [x, y, seat], [0, 0, seat / 2], FABRIC, mu=0.9),
                Part('back', 'box', [x, 0.20, z - seat], [0, -y / 2 + 0.10, seat + (z - seat) / 2], FABRIC),
                Part('arm_l', 'box', [0.18, y, 0.65], [-x / 2 + 0.09, 0, 0.325], FABRIC),
                Part('arm_r', 'box', [0.18, y, 0.65], [x / 2 - 0.09, 0, 0.325], FABRIC)]
    if kind == 'fridge':
        return [Part('body', 'box', s, [0, 0, z / 2], (0.80, 0.82, 0.85)),
                Part('handle', 'box', [0.02, 0.03, 0.6], [-x / 2 - 0.01, -y / 4, z * 0.6], STEEL, collide=False)]
    if kind == 'counter':
        return [Part('body', 'box', [x, y, z - 0.04], [0, 0, (z - 0.04) / 2], DARK),
                Part('top', 'box', [x, y, 0.04], [0, 0, z - 0.02], (0.30, 0.30, 0.32), mu=0.8)]
    if kind == 'sink':
        return [Part('body', 'box', [x, y, z - 0.04], [0, 0, (z - 0.04) / 2], DARK),
                Part('top', 'box', [x, y, 0.04], [0, 0, z - 0.02], STEEL, mu=0.8),
                Part('basin', 'box', [x * 0.7, y * 0.6, 0.01], [0, 0, z + 0.001], (0.4, 0.42, 0.45), collide=False),
                Part('faucet', 'cylinder', [0.03, 0.03, 0.25], [0, y / 2 - 0.06, z + 0.125], STEEL)]
    if kind == 'stove':
        p = [Part('body', 'box', [x, y, z - 0.04], [0, 0, (z - 0.04) / 2], WHITE),
             Part('top', 'box', [x, y, 0.04], [0, 0, z - 0.02], (0.1, 0.1, 0.1), mu=0.8)]
        for i, (bx, by) in enumerate([(-0.12, -0.13), (0.12, -0.13), (-0.12, 0.13), (0.12, 0.13)]):
            p.append(Part(f'burner{i}', 'cylinder', [0.16, 0.16, 0.005], [bx, by, z + 0.003], (0.25, 0.25, 0.25), collide=False))
        return p
    raise ValueError(kind)


# Sillas de la mesa del comedor. El plano 10 muestra sillas alrededor; se dejan 2 (este y norte): con una silla al
# oeste el robot (radio de giro 0.43 m) no puede girar tras cruzar la puerta sala-cocina (alineada con la mesa) y el
# lado sur queda para la demo de Pick and Place. Decisión de simulación, no dato del plano.
CHAIR_ANGLES = (0.0, math.pi / 2)


def table_chairs(fu):
    """(nombre, xy, yaw, partes) de cada silla de una mesa redonda (fuente única para el SDF y el mapa 2D)."""
    out = []
    for k, ang in enumerate(CHAIR_ANGLES):
        cx, cy = fu['xy'][0] + 0.75 * math.cos(ang), fu['xy'][1] + 0.75 * math.sin(ang)
        out.append((f'{fu["name"]}_chair{k}', [cx, cy], ang + math.pi, [
            Part('seat', 'box', [0.42, 0.42, 0.04], [0, 0, 0.45], DARK),
            Part('back', 'box', [0.04, 0.42, 0.45], [-0.19, 0, 0.70], DARK),
            Part('legs', 'box', [0.36, 0.36, 0.43], [0, 0, 0.215], DARK)]))
    return out


# ---------------------------------------------------------------------------------------------- mundo
def world_header(name, step):
    return f"""<?xml version="1.0"?>
<!-- GENERADO por solverbot_sim/tools/make_worlds.py: no editar a mano. Gazebo Harmonic (gz-sim 8). -->
<sdf version="1.10">
  <world name="{name}">
    <physics name="default_physics" default="true" type="dart">
      <max_step_size>{step}</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <scene><ambient>0.6 0.6 0.6 1</ambient><background>0.75 0.8 0.9 1</background><grid>false</grid></scene>
    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows><pose>0 0 10 0 0 0</pose>
      <diffuse>0.85 0.85 0.85 1</diffuse><specular>0.2 0.2 0.2 1</specular><direction>-0.4 0.2 -0.9</direction>
    </light>
    <model name="ground_plane"><static>true</static><link name="link">
      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>60 60</size></plane></geometry>
        <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision>
      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>60 60</size></plane></geometry>{material((0.55, 0.57, 0.55))}</visual>
    </link></model>
"""


def walls(spec):
    h = spec['house']
    W, L = h['size']
    t, hh, dh = h['wall_thickness'], h['wall_height'], h['door_height']
    ix, iy = h['interior_wall_x'], h['interior_wall_y']
    # Cada pared: (nombre, eje 'x'|'y', coordenada fija, desde, hasta)
    defs = {'south': ('y', t / 2, 0, W), 'north': ('y', L - t / 2, 0, W), 'west': ('x', t / 2, 0, L),
            'east': ('x', W - t / 2, 0, L), 'interior_x': ('x', ix, t, L - t), 'interior_y': ('y', iy, t, W - t)}
    parts = []
    for wname, (axis, c, a, b) in defs.items():
        gaps = sorted((o['center'] - o['width'] / 2, o['center'] + o['width'] / 2)
                      for o in spec['openings'] if o['wall'] == wname)
        segs, cur = [], a
        for g0, g1 in gaps:
            segs.append((cur, g0))
            # dintel sobre el vano (altura de la puerta de referencia)
            mid, ln = (g0 + g1) / 2, g1 - g0
            xyz = [c, mid, dh + (hh - dh) / 2] if axis == 'x' else [mid, c, dh + (hh - dh) / 2]
            size = [t, ln, hh - dh] if axis == 'x' else [ln, t, hh - dh]
            parts.append(Part(f'{wname}_lintel_{len(parts)}', 'box', size, xyz, WHITE))
            cur = g1
        segs.append((cur, b))
        for i, (s0, s1) in enumerate(segs):
            if s1 - s0 < 1e-3:
                continue
            mid, ln = (s0 + s1) / 2, s1 - s0
            xyz = [c, mid, hh / 2] if axis == 'x' else [mid, c, hh / 2]
            size = [t, ln, hh] if axis == 'x' else [ln, t, hh]
            parts.append(Part(f'{wname}_{i}', 'box', size, xyz, (0.96, 0.95, 0.92)))
    return static_model('house_walls', [0, 0], 0, parts)


def floors(spec):
    parts = []
    for name, r in spec['rooms'].items():
        (x0, y0), (x1, y1) = r['min'], r['max']
        parts.append(Part(name, 'box', [x1 - x0, y1 - y0, 0.005], [(x0 + x1) / 2, (y0 + y1) / 2, 0.0025],
                          r['floor_rgb'], collide=False))
    return static_model('room_floors', [0, 0], 0, parts)


def home_arena(spec):
    out = [world_header('home_arena', 0.001), walls(spec), floors(spec)]
    top = {}
    for fu in spec['furniture']:
        out.append(static_model(fu['name'], fu['xy'], fu.get('yaw', 0.0), furniture_parts(fu['kind'], fu['size'])))
        z = fu['size'][2]
        # superficie de apoyo: tapa del mueble, o el segundo estante en estanterías/armarios abiertos
        top[fu['name']] = (fu['xy'], 0.03 + (z - 0.04) / 3 if fu['kind'] in ('shelf', 'cabinet_open') else z)
        if fu['kind'] == 'round_table':
            for name, xy, yaw, parts in table_chairs(fu):
                out.append(static_model(name, xy, yaw, parts))
    for o in spec['objects']:
        (fx, fy), h = top[o['place_on']]
        z = h + o['size'][2] / 2 + 0.005
        out.append(dynamic_object(o['name'], o['shape'], o['size'], o['mass'], o['rgb'],
                                  [fx + o['xy_off'][0], fy + o['xy_off'][1], z]))
    out.append('  </world>\n</sdf>\n')
    return ''.join(out)


def training_world(task):
    tb, ob, tz = task['table'], task['object'], task['target_zone']
    out = [world_header('pick_place_training', task['physics_step'])]
    out.append(static_model('training_table', tb['xy'], 0.0, furniture_parts('table', tb['size'])))
    # Zona objetivo: marca SOLO visual (sin colisión) apoyada en la mesa
    out.append(static_model('target_zone', tz['xy'], 0.0, [
        Part('mark', 'box', [tz['size'][0], tz['size'][1], 0.002], [0, 0, tb['size'][2] + 0.001], (0.1, 0.8, 0.2),
             collide=False)]))
    z = tb['size'][2] + ob['size'][2] / 2 + 0.002
    out.append(dynamic_object(ob['name'], ob['shape'], ob['size'], ob['mass'], ob['rgb'],
                              [ob['xy'][0], ob['xy'][1], z], mu=ob.get('mu', 1.0), publish_pose=True))
    out.append('  </world>\n</sdf>\n')
    return ''.join(out)


def main():
    with open(os.path.join(PKG, 'config', 'arena_spec.yaml')) as fh:
        spec = yaml.safe_load(fh)
    with open(os.path.join(PKG, 'config', 'pick_place.yaml')) as fh:
        task = yaml.safe_load(fh)['world']
    os.makedirs(os.path.join(PKG, 'worlds'), exist_ok=True)
    for name, text in (('home_arena.sdf', home_arena(spec)), ('pick_place_training.sdf', training_world(task))):
        with open(os.path.join(PKG, 'worlds', name), 'w') as fh:
            fh.write(text)
        print('escrito', name)
    return 0


if __name__ == '__main__':
    sys.exit(main())
