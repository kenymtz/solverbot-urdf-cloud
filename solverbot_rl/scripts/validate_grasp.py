#!/usr/bin/env python3
"""Validación FÍSICA del agarre/levantamiento con el experto guionado, por semilla, en Gazebo (o cinemático).

Criterio de éxito físico (independiente de la recompensa): el cubo REAL queda >= --min-rise (30 mm) por encima de
su altura apoyada durante >= --hold-steps pasos consecutivos (10 = 1 s simulado), a < 4 cm del TCP. Se informa además
el error de mimic |joint7 + joint8|, la deriva XY del cubo antes del cierre y la apertura al sostener.

    # simulación en marcha (tools/headless/start_sim.sh o tools/wsl/run_sim.sh), mismo ROS_DOMAIN_ID / GZ_PARTITION
    python3 solverbot_rl/scripts/validate_grasp.py --backend gazebo --seeds 40000-40004 --out runs/grasp_val

Salida: <out>/steps.csv (una fila por paso), <out>/summary.json (por semilla + tasa con denominador).
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solverbot_rl import task as T  # noqa: E402
from solverbot_rl.envs.pick_place import SolverBotPickPlaceEnv  # noqa: E402
from solverbot_rl.expert import ScriptedExpert, rotation_error_deg  # noqa: E402,F401


def parse_seeds(text):
    out = []
    for part in text.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def run_seed(env, seed, steps, min_rise, hold_steps, writer):
    ex = ScriptedExpert(env)
    env.reset(seed=seed)
    ex.reset()
    e = env.unwrapped
    rest_z = e._state.obj_pos[2]
    obj0 = e._state.obj_pos.copy()
    yaw0 = float(e.backend.object_yaw_truth())
    rec = {'seed': seed, 'cube_xy0': obj0[:2].round(4).tolist(), 'cube_yaw0_deg': round(np.degrees(yaw0), 1),
           'max_mimic_err_mm': 0.0, 'drift_before_close_mm': None, 'q7_at_block_mm': None, 'max_rise_mm': 0.0,
           'best_hold_steps': 0, 'final_rise_mm': None, 'final_q7_mm': None, 'final_q8_mm': None,
           'grasp_tilt_deg': None, 'phases': [], 'collision': False, 'dropped': False}
    hold = 0
    t0 = time.time()
    for k in range(steps):
        a = ex.act()
        if ex.R is not None and rec['grasp_tilt_deg'] is None:
            rec['grasp_tilt_deg'] = round(float(np.degrees(np.arccos(np.clip(-ex.R[2, 2], -1, 1)))), 1)
        _, _, term, trunc, info = env.step(a)
        s = e._state
        rise = float(s.obj_pos[2] - rest_z)
        held = np.linalg.norm(s.obj_pos - s.tcp_pos) < 0.04
        hold = hold + 1 if (rise >= min_rise and held) else 0
        mim = T.mimic_error(s) * 1000
        rec['max_mimic_err_mm'] = max(rec['max_mimic_err_mm'], round(mim, 3))
        rec['max_rise_mm'] = max(rec['max_rise_mm'], round(rise * 1000, 2))
        rec['best_hold_steps'] = max(rec['best_hold_steps'], hold)
        if ex.phase in ('close', 'squeeze') and rec['drift_before_close_mm'] is None:
            rec['drift_before_close_mm'] = round(float(np.linalg.norm(s.obj_pos[:2] - obj0[:2])) * 1000, 2)
        if ex.phase == 'squeeze' and rec['q7_at_block_mm'] is None:
            rec['q7_at_block_mm'] = round(s.gripper_q * 1000, 2)
        if not rec['phases'] or rec['phases'][-1] != ex.phase:
            rec['phases'].append(ex.phase)
        rec['collision'] |= 'pen_collision' in info['terms']
        rec['dropped'] |= 'pen_drop' in info['terms']
        writer.writerow({'seed': seed, 'step': k, 'phase': ex.phase, 'q7_mm': round(s.gripper_q * 1000, 3),
                         'q8_mm': round((s.gripper_q8 or 0.0) * 1000, 3), 'mimic_err_mm': round(mim, 3),
                         'grip_target_mm': round(s.gripper_target * 1000, 3),
                         'tcp_x': round(s.tcp_pos[0], 4), 'tcp_y': round(s.tcp_pos[1], 4), 'tcp_z': round(s.tcp_pos[2], 4),
                         'cube_x': round(s.obj_pos[0], 4), 'cube_y': round(s.obj_pos[1], 4), 'cube_z': round(s.obj_pos[2], 4),
                         'cube_rise_mm': round(rise * 1000, 2), 'hold_steps': hold})
        if rec['collision'] or rec['dropped'] or ex.phase == 'failed':
            break
        if hold >= hold_steps + 5:            # ya demostrado con margen
            break
    s = e._state
    rec.update({'final_rise_mm': round(float(s.obj_pos[2] - rest_z) * 1000, 2), 'final_q7_mm': round(s.gripper_q * 1000, 2),
                'final_q8_mm': round((s.gripper_q8 or 0.0) * 1000, 2), 'steps': k + 1,
                'wall_s': round(time.time() - t0, 1)})
    rec['success'] = bool(rec['best_hold_steps'] >= hold_steps and not rec['collision'] and not rec['dropped'])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['gazebo', 'kinematic'], default='gazebo')
    ap.add_argument('--seeds', default='40000-40004')
    ap.add_argument('--steps', type=int, default=160)
    ap.add_argument('--min-rise', type=float, default=0.030)
    ap.add_argument('--hold-steps', type=int, default=10)
    ap.add_argument('--level', type=int, default=4, help='4 = pick (desde la pose inicial), 3 = lift')
    ap.add_argument('--set', action='append', default=[], help='override de config, p. ej. task.grasp_dz=-0.004')
    ap.add_argument('--out', default='runs/grasp_val')
    args = ap.parse_args()
    overrides = {'control.max_episode_steps': 10_000, 'task.no_progress_steps': 10_000,
                 'observation.object_source': 'privileged'}
    for kv in args.set:
        k, v = kv.split('=', 1)
        overrides[k] = json.loads(v)
    os.makedirs(args.out, exist_ok=True)
    env = SolverBotPickPlaceEnv(backend=args.backend, level=args.level, overrides=overrides)
    rows = []
    with open(os.path.join(args.out, 'steps.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['seed', 'step', 'phase', 'q7_mm', 'q8_mm', 'mimic_err_mm', 'grip_target_mm',
                                           'tcp_x', 'tcp_y', 'tcp_z', 'cube_x', 'cube_y', 'cube_z', 'cube_rise_mm',
                                           'hold_steps'])
        w.writeheader()
        for seed in parse_seeds(args.seeds):
            rec = run_seed(env, seed, args.steps, args.min_rise, args.hold_steps, w)
            fh.flush()
            rows.append(rec)
            print(json.dumps({k: rec[k] for k in ('seed', 'success', 'max_rise_mm', 'best_hold_steps', 'final_q7_mm',
                                                  'final_q8_mm', 'max_mimic_err_mm', 'drift_before_close_mm',
                                                  'q7_at_block_mm', 'grasp_tilt_deg', 'phases')}), flush=True)
    n_ok = sum(r['success'] for r in rows)
    summary = {'backend': args.backend, 'physical': env.unwrapped.backend.physical, 'level': args.level,
               'criterion': f'rise >= {args.min_rise * 1000:.0f} mm during >= {args.hold_steps} steps, cube < 4 cm from TCP',
               'overrides': overrides, 'successes': n_ok, 'episodes': len(rows),
               'success_rate': n_ok / max(1, len(rows)), 'seeds': rows}
    with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    print(f'RESULTADO: {n_ok}/{len(rows)} levantamientos físicos sostenidos ({args.backend})')
    env.close()
    return 0 if n_ok == len(rows) else 1


if __name__ == '__main__':
    sys.exit(main())
