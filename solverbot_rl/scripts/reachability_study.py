#!/usr/bin/env python3
"""Estudio de alcanzabilidad del brazo (Doing Laundry): ¿qué puntos del cesto, del tambor del lavarropas y de la mesa
de doblado puede alcanzar la pinza con las articulaciones y límites REALES del URDF (incluido el elevador
torso_lift_joint en [-0.19, 0.068] m), sin colisiones groseras con el piso, la base o la pared del cesto?

Para cada punto objetivo (en el frame base_footprint) se resuelve IK con restarts, con el elevador como articulación
libre, y tres restricciones de orientación de la pinza:
  down       eje de aproximación hacia abajo (tomar una prenda desde arriba, dentro de un cesto)
  horizontal eje de aproximación horizontal hacia +y de la base (entrar por la boca del tambor, lado del brazo)
  libre      solo posición
Un punto es alcanzable si el error de posición < 1 cm, el de orientación < 15° y ningún punto de los eslabones del
brazo (orígenes y puntos medios) queda bajo el piso (z < 1 cm) o dentro del prisma de la base (z < 0.42 m).

    python3 solverbot_rl/scripts/reachability_study.py --out docs/resultados/desafios/alcance_brazo.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solverbot_rl.config import load_urdf  # noqa: E402
from solverbot_rl.kinematics import UrdfChain, solve_ik  # noqa: E402
from solverbot_rl.challenges.arena import FOOTPRINT  # noqa: E402
from solverbot_rl.challenges.base_env import STOW  # noqa: E402

ARM = [f'joint{i}' for i in range(1, 7)]
JOINTS = ARM + ['torso_lift_joint']
LINKS = ['link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'gripper_base', 'tcp_link']
AXES = {'down': np.array([0.0, 0.0, -1.0]), 'horizontal': np.array([0.0, 1.0, 0.0]), 'libre': None}


def check(chain, q, target, axis, extra_obstacle=None):
    js = dict(zip(JOINTS, q))
    T = chain.link_pose('tcp_link', js)
    pos_err = float(np.linalg.norm(T[:3, 3] - target))
    ang = 0.0 if axis is None else math.degrees(math.acos(np.clip(T[:3, 2] @ axis, -1, 1)))
    pts = [chain.link_pose(l, js)[:3, 3] for l in LINKS]
    pts += [(a + b) / 2 for a, b in zip(pts[:-1], pts[1:])]
    pts = np.array(pts)
    x0, x1, y0, y1 = FOOTPRINT
    floor = bool(np.any(pts[:, 2] < 0.01))
    in_base = bool(np.any((pts[:, 2] < 0.42) & (pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)))
    wall = bool(extra_obstacle(pts)) if extra_obstacle else False
    return pos_err, ang, floor, in_base, wall


def reach(chain, target, axis_name, lift_fixed=None, extra_obstacle=None, restarts=6):
    axis = AXES[axis_name]
    fixed = {'torso_lift_joint': lift_fixed} if lift_fixed is not None else {}
    joints = ARM if lift_fixed is not None else JOINTS
    q0 = np.array(list(STOW) + ([] if lift_fixed is not None else [0.0]))
    kw = {} if axis is None else {'approach_axis': axis}
    q, err = solve_ik(chain, 'tcp_link', joints, q0, np.asarray(target, float), None, fixed, restarts=restarts,
                      reg=0.001, **kw)
    full = list(q) + ([lift_fixed] if lift_fixed is not None else [])
    pos_err, ang, floor, in_base, wall = check(chain, full, np.asarray(target, float), axis, extra_obstacle)
    ok = pos_err < 0.01 and ang < 15 and not floor and not in_base and not wall
    return ok, {'pos_err_mm': round(pos_err * 1000, 1), 'ang_deg': round(ang, 1), 'floor': floor, 'in_base': in_base,
                'wall': wall, 'lift': round(float(full[-1]), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    chain = UrdfChain(load_urdf())
    t0 = time.time()
    rep = {'joint_limits': {j: chain.limits([j]) for j in JOINTS} if hasattr(chain, 'limits') else None}
    try:
        lo, hi = chain.limits(JOINTS)
        rep['joint_limits'] = {j: [round(float(a), 4), round(float(b), 4)] for j, a, b in zip(JOINTS, lo, hi)}
    except Exception:
        pass

    # 1) Mapa: z mínima alcanzable (pinza hacia abajo) por distancia lateral y (lado del brazo), mejor x en {0, .1, .2}
    ys = np.arange(0.35, 0.86, 0.05)
    zs = np.arange(0.02, 0.81, 0.04) if not args.quick else np.arange(0.02, 0.81, 0.08)
    zmin = {}
    for mode, lift in (('elevador_libre', None), ('elevador_0', 0.0)):
        row = {}
        for y in ys:
            best = None
            for z in zs:
                if any(reach(chain, (x, y, z), 'down', lift)[0] for x in (0.1, 0.0, 0.2)):
                    best = round(float(z), 2)
                    break
            row[round(float(y), 2)] = best
        zmin[mode] = row
        print('[mapa]', mode, row, flush=True)
    rep['zmin_down_por_y'] = zmin

    # 2) Cesto de piso (challenges.yaml: 0.45 x 0.36 x 0.30), robot de costado: centro del cesto a y = 0.62 (mínimo
    #    geométrico 0.58: medio ancho 0.18 + huella izquierda 0.363 + holgura). Pared cercana del cesto en y = 0.44.
    bw = (0.62 - 0.18, 0.62 + 0.18)

    def basket_wall(pts):
        # puntos del brazo por debajo del borde (z < 0.30) fuera del interior del cesto en y -> chocan con la pared
        low = pts[pts[:, 2] < 0.30]
        return np.any((low[:, 1] < bw[0] + 0.01) | (low[:, 1] > bw[1] - 0.01)) if len(low) else False
    basket = {}
    for depth in (0.05, 0.10, 0.20, 0.30):            # altura de la prenda sobre el piso (fondo 0.015; pila de prendas)
        for dy in (-0.10, 0.0, 0.10):
            tgt = (0.1, 0.62 + dy, depth + 0.02)
            ok, d = reach(chain, tgt, 'down', None, basket_wall)
            basket[f'z={depth:.2f} y={0.62 + dy:.2f}'] = {'ok': ok, **d}
    rep['cesto_piso'] = basket
    print('[cesto]', {k: v['ok'] for k, v in basket.items()}, flush=True)

    # 3) Tambor de lavarropas frontal (make_worlds: cuerpo 0.60 x 0.60 x 0.85, puerta circular de 0.36 m centrada a
    #    z = 0.38). Robot de costado con el frente del lavarropas a y = 0.41 (huella 0.363 + 5 cm). Puntos dentro del
    #    tambor: boca (y 0.41) a 0.30 m de profundidad, z en la boca.
    #    Todo punto del brazo que entra al lavarropas (y > 0.41) debe pasar por la boca: radio 0.18 - 0.03 (grosor
    #    aproximado de los eslabones, ~6 cm de diámetro) alrededor del eje de la puerta (x = 0.1, z = 0.38).
    def drum_mouth(pts):
        inside = pts[pts[:, 1] > 0.41]
        return bool(np.any(np.hypot(inside[:, 0] - 0.1, inside[:, 2] - 0.38) > 0.15)) if len(inside) else False
    drum = {}
    for depth in (0.0, 0.10, 0.20, 0.30):
        for z in (0.26, 0.38, 0.50):
            tgt = (0.1, 0.41 + depth, z)
            ok, d = reach(chain, tgt, 'horizontal', None, drum_mouth)
            drum[f'prof={depth:.2f} z={z:.2f}'] = {'ok': ok, **d}
    rep['tambor_lavarropas'] = drum
    print('[tambor]', {k: v['ok'] for k, v in drum.items()}, flush=True)

    # 4) Mesa de doblado (0.90 x 0.56 x 0.74): superficie a z = 0.74, robot de costado con el borde a y = 0.41
    table = {}
    for depth in (0.05, 0.15, 0.25, 0.35, 0.45):
        for x in (-0.1, 0.1, 0.3):
            ok, d = reach(chain, (x, 0.41 + depth, 0.76), 'down', None)
            table[f'x={x:.1f} prof={depth:.2f}'] = {'ok': ok, **d}
    rep['mesa_doblado'] = table
    print('[mesa]', {k: v['ok'] for k, v in table.items()}, flush=True)
    rep['wall_s'] = round(time.time() - t0, 1)
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(rep, fh, indent=1)


if __name__ == '__main__':
    main()
