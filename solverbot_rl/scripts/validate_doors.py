#!/usr/bin/env python3
"""Validación física del paso por puertas (riesgo crítico: ~15.5 mm de holgura por lado con la puerta de 0.70 m).

Recorre una ruta que cruza las 4 puertas transitables en ambos sentidos con la navegación de producción
(skill_navigate: pose ESTIMADA por LiDAR + odometría, pausa de convergencia, cruce recto) y registra, en cada cruce,
la holgura geométrica EXACTA mínima (huella real contra las primitivas de colisión del mundo) y el error lateral y de
rumbo reales respecto del eje de la puerta. No toca la geometría: ni puertas más anchas ni robot más chico.

    # simulación del desafío en marcha (world:=home_challenges cameras:=true), mismo ROS_DOMAIN_ID / GZ_PARTITION
    python3 solverbot_rl/scripts/validate_doors.py --backend gazebo --loops 2 --out docs/resultados/desafios/puertas_gz.json
    python3 solverbot_rl/scripts/validate_doors.py --backend kinematic --loops 3 --seed 0
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solverbot_rl.challenges.laundry_env import LaundryEnv  # noqa: E402
from solverbot_rl.challenges.arena import FOOTPRINT  # noqa: E402

FC = ((FOOTPRINT[0] + FOOTPRINT[1]) / 2, (FOOTPRINT[2] + FOOTPRINT[3]) / 2)

# ruta: afuera -> sala -> cocina -> dormitorio -> cocina -> sala -> lavadero -> sala (cruza 4 puertas, ida y vuelta)
ROUTE = ['living_room', 'kitchen', 'bedroom', 'kitchen', 'living_room', 'laundry', 'living_room']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['kinematic', 'gazebo'], default='kinematic')
    ap.add_argument('--loops', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    env = LaundryEnv(backend=args.backend, level=0)       # sin personas en escena; solo la casa y la puerta de entrada
    env.reset(seed=args.seed)
    env.open_entrance_door()
    a = env.arena
    doors = [d for d in a.doors if d.usable]
    crossings, cur = [], {}
    orig = env.sim_step

    def step(*k, **kw):
        orig(*k, **kw)
        tr = env.truth()
        c, s_ = math.cos(tr[2]), math.sin(tr[2])
        fc = tr[:2] + np.array([c * FC[0] - s_ * FC[1], s_ * FC[0] + c * FC[1]])   # centro de la huella
        for d in doors:
            rel = fc - d.center
            along, lat = float(rel @ d.normal), float(rel @ np.array([-d.normal[1], d.normal[0]]))
            inside = abs(along) < 0.35 and abs(lat) < 0.3          # algún punto de la huella en el vano
            if inside:
                clr, who = a.footprint_clearance(tr[0], tr[1], tr[2])
                yaw_err = math.degrees(math.atan2(math.sin(tr[2] - math.atan2(d.normal[1], d.normal[0])),
                                                  math.cos(tr[2] - math.atan2(d.normal[1], d.normal[0]))))
                yaw_err = (yaw_err + 90) % 180 - 90                  # cruce en cualquiera de los dos sentidos
                c = cur.setdefault(d.name, {'door': d.name, 'min_clearance_mm': 1e9, 'max_abs_lat_mm': 0.0,
                                            'max_abs_yaw_deg': 0.0, 'loc_err_mm': [], 't0': env.sim_time})
                c['min_clearance_mm'] = min(c['min_clearance_mm'], clr * 1000)
                c['max_abs_lat_mm'] = max(c['max_abs_lat_mm'], abs(lat) * 1000)   # 0 = huella centrada
                c['max_abs_yaw_deg'] = max(c['max_abs_yaw_deg'], abs(yaw_err))
                c['loc_err_mm'].append(float(np.linalg.norm(env.est[:2] - tr[:2])) * 1000)
            elif d.name in cur:
                c = cur.pop(d.name)
                c['loc_err_mean_mm'] = float(np.mean(c.pop('loc_err_mm')))
                c['dur_s'] = round(env.sim_time - c.pop('t0'), 1)
                c['contact'] = bool(env.contact)
                crossings.append({k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items()})
                print('[puerta]', crossings[-1], flush=True)

    env.sim_step = step
    legs, t0 = [], time.time()
    route = ROUTE * args.loops
    for goal in route:
        loc = a.locations[goal]
        res = env.skill_navigate((loc.x, loc.y), None)
        legs.append({'goal': goal, 'ok': res.ok, 'reason': res.get('reason'), 'sim_time': round(env.sim_time, 1)})
        print('[tramo]', legs[-1], flush=True)
        if env.failed():
            break
    env.sim_step = orig
    info = env.common_info()
    rep = {'backend': args.backend, 'physical': args.backend == 'gazebo', 'loops': args.loops, 'seed': args.seed,
           'route': route, 'legs': legs, 'crossings': crossings,
           'summary': {'legs_ok': sum(l['ok'] for l in legs), 'legs': len(legs), 'crossings': len(crossings),
                       'contacts': int(info['contact']), 'contact_with': info.get('contact_with'),
                       'min_clearance_mm': min((c['min_clearance_mm'] for c in crossings), default=None),
                       'max_abs_lat_mm': max((c['max_abs_lat_mm'] for c in crossings), default=None),
                       'max_abs_yaw_deg': max((c['max_abs_yaw_deg'] for c in crossings), default=None),
                       'loc_err_mean_m': info['loc_err_mean'], 'loc_err_max_m': info['loc_err_max'],
                       'safety_halts': info.get('safety_halts'), 'wall_s': round(time.time() - t0, 1)}}
    print(json.dumps(rep['summary'], indent=1))
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(rep, fh, indent=1)
    env.close()


if __name__ == '__main__':
    main()
