#!/usr/bin/env python3
"""Evaluación autónoma de políticas en `reach` (y niveles experimentales) con métricas físicas por episodio.

    python3 solverbot_rl/scripts/eval_reach.py --backend gazebo --policy runs/reach_bc_v1/bc_v1/model.zip \
        --level reach_var_m --seeds 900000-900029 --out runs/reach_bc_v1/eval/bc_v1_var_m

--policy: ruta .zip (SB3 SAC: BC o fine-tuning) | scripted (experto, referencia) | random | zero.
Con un modelo, el experto guionado NO se instancia: las acciones provienen solo de model.predict(obs) con la
observación de Gymnasium (object_source del entorno, perception por defecto).

Salida: <out>/summary.json (agregados + IC de Wilson 95 %), <out>/index.jsonl y <out>/episodes/*.npz (trazas).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solverbot_rl.envs.pick_place import SolverBotPickPlaceEnv  # noqa: E402
from solverbot_rl.imitation import dataset as D  # noqa: E402
from solverbot_rl.imitation.experiment import env_kwargs, level_index, load_experiment, parse_seeds  # noqa: E402
from solverbot_rl.imitation.rollout import run_episode  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def aggregate(rows):
    n = len(rows)
    k = sum(r['success'] for r in rows)

    def m(key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return {'mean': float(np.mean(v)), 'median': float(np.median(v)), 'min': float(np.min(v)),
                'max': float(np.max(v))} if v else None

    succ = [r for r in rows if r['success']]
    return {'episodes': n, 'successes': int(k), 'success_rate': k / n if n else 0.0, 'success_wilson95': wilson(k, n),
            'd0_mm': m('d0_mm'), 'dmin_mm': m('dmin_mm'), 'dfinal_mm': m('dfinal_mm'),
            'improvement_mm': {'mean': float(np.mean([r['d0_mm'] - r['dmin_mm'] for r in rows]))},
            'tcp_z2_final': m('tcp_z2_final'), 'return': m('return'), 'length': m('length'),
            'length_success': float(np.mean([r['length'] for r in succ])) if succ else None,
            'collision_rate': float(np.mean([r['collision'] for r in rows])),
            'stalled_rate': float(np.mean([r['stalled'] for r in rows])),
            'timeout_rate': float(np.mean([(not r['success']) and (not r['stalled']) and (not r['collision'])
                                           for r in rows])),
            'detection_valid_rate': float(np.mean([r['valid_rate'] for r in rows])),
            'episodes_without_any_detection': int(sum(r['valid_rate'] == 0 for r in rows)),
            'perc_err_mean_mm': m('perc_err_mean_mm'),
            'stale_images_total': int(sum(r['stale_images_total'] for r in rows)),
            'failed_seeds': [r['seed'] for r in rows if not r['success']]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--backend', choices=['kinematic', 'gazebo'], default='gazebo')
    ap.add_argument('--experiment', default='reach_bc_v1')
    ap.add_argument('--level', default='reach')
    ap.add_argument('--seeds', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stochastic', action='store_true', help='muestrear de la política (por defecto determinista)')
    args = ap.parse_args()

    exp = load_experiment(args.experiment)
    env = SolverBotPickPlaceEnv(backend=args.backend, **env_kwargs(exp))
    lvl = level_index(env, args.level)
    expert, act = None, None
    if args.policy == 'scripted':
        from solverbot_rl.expert import ScriptedExpert
        expert = ScriptedExpert(env)
    elif args.policy == 'random':
        rng = np.random.default_rng(12345)

        def act(_):
            return rng.uniform(-1, 1, 7).astype(np.float32)
    elif args.policy == 'zero':
        def act(_):
            return np.zeros(7, np.float32)
    else:
        from solverbot_rl.imitation.bc import load_policy
        act = load_policy(args.policy, deterministic=not args.stochastic)

    os.makedirs(args.out, exist_ok=True)
    done = {r['seed'] for r in D.read_index(args.out)}
    seeds = [s for s in parse_seeds(args.seeds) if s not in done]
    print(f'[eval] policy={args.policy} nivel {lvl}:{env.levels[lvl]["name"]} backend={args.backend} '
          f'object_source={env.object_source} experto_instanciado={expert is not None} '
          f'{len(seeds)} episodios ({len(done)} previos)', flush=True)
    for i, seed in enumerate(seeds):
        arrays, s = run_episode(env, seed, lvl, act=act, expert=expert, rng=np.random.default_rng(seed))
        s.update({'policy': args.policy, 'expert_in_loop': expert is not None})
        D.save_episode(args.out, arrays, s)
        print(f'[eval] {i + 1}/{len(seeds)} seed={seed} éxito={s["success"]} pasos={s["length"]} '
              f'd0={s["d0_mm"]:.1f} dmin={s["dmin_mm"]:.1f} dfin={s["dfinal_mm"]:.1f} mm zf={s["tcp_z2_final"]:.3f} '
              f'R={s["return"]:.2f} col={s["collision"]} stall={s["stalled"]} det={s["valid_rate"]:.2f}', flush=True)
    rows = sorted(D.read_index(args.out), key=lambda r: r['seed'])
    summ = {'policy': args.policy, 'backend': args.backend, 'physical': env.backend.physical,
            'experiment': exp['name'], 'level': lvl, 'level_def': env.levels[lvl],
            'object_source': env.object_source, 'deterministic': not args.stochastic,
            'expert_in_loop': expert is not None, 'seeds': args.seeds, **aggregate(rows)}
    with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps({k: summ[k] for k in ('episodes', 'successes', 'success_rate', 'success_wilson95',
                                            'collision_rate', 'stalled_rate')}), flush=True)
    env.close()


if __name__ == '__main__':
    main()
