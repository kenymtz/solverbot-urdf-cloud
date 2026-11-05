#!/usr/bin/env python3
"""Fase F: evaluación de la política congelada CON el guardián de compromiso de rama (imitation/branch_guard.py).

Misma interfaz y salidas que eval_reach.py; el guardián solo modifica la acción de joint4 cuando empuja hacia la otra
rama después del compromiso. Sin experto, sin verdad del simulador en la política.

    python3 solverbot_rl/scripts/eval_branch_guard.py --backend kinematic \
        --policy solverbot_rl/docs/resultados/reach_bc_v1/modelos/bc_v2w.zip --level reach_var_m \
        --seeds 904000-904199 --out runs/dual_camera/guard/kin_bc_v2w_var_m
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solverbot_rl.envs.pick_place import SolverBotPickPlaceEnv  # noqa: E402
from solverbot_rl.imitation import dataset as D  # noqa: E402
from solverbot_rl.imitation.branch_guard import BranchCommitGuard  # noqa: E402
from solverbot_rl.imitation.experiment import env_kwargs, level_index, load_experiment, parse_seeds  # noqa: E402
from solverbot_rl.imitation.rollout import run_episode  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_reach import aggregate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--backend', choices=['kinematic', 'gazebo'], default='gazebo')
    ap.add_argument('--experiment', default='reach_bc_v1')
    ap.add_argument('--level', default='reach_var_m')
    ap.add_argument('--seeds', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--commit', type=float, default=0.15)
    ap.add_argument('--max-reverse', type=float, default=0.2)
    args = ap.parse_args()
    from solverbot_rl.imitation.bc import load_policy

    env = SolverBotPickPlaceEnv(backend=args.backend, **env_kwargs(load_experiment(args.experiment)))
    lvl = level_index(env, args.level)
    pol = load_policy(args.policy, deterministic=True)
    guard = BranchCommitGuard(env.lo, env.hi, args.commit, args.max_reverse)
    os.makedirs(args.out, exist_ok=True)
    done = {r['seed'] for r in D.read_index(args.out)}
    for seed in [s for s in parse_seeds(args.seeds) if s not in done]:
        guard.reset()
        arrays, s = run_episode(env, seed, lvl, act=lambda o: guard(o, pol(o)), rng=np.random.default_rng(seed))
        s.update({'policy': args.policy, 'expert_in_loop': False, 'branch_guard': guard.summary(),
                  'guard_params': {'commit': args.commit, 'max_reverse': args.max_reverse}})
        D.save_episode(args.out, arrays, s)
        print(f'[guard] seed={seed} éxito={s["success"]} pasos={s["length"]} dmin={s["dmin_mm"]:.1f} '
              f'dfin={s["dfinal_mm"]:.1f} zf={s["tcp_z2_final"]:.3f} {guard.summary()}', flush=True)
    rows = sorted(D.read_index(args.out), key=lambda r: r['seed'])
    summ = {'policy': args.policy, 'backend': args.backend, 'level': args.level, 'seeds': args.seeds,
            'branch_guard': {'commit': args.commit, 'max_reverse': args.max_reverse}, **aggregate(rows),
            'guard_interventions_mean': float(np.mean([r['branch_guard']['interventions'] for r in rows]))}
    with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps({k: summ[k] for k in ('episodes', 'successes', 'success_wilson95', 'failed_seeds',
                                            'guard_interventions_mean')}), flush=True)
    env.close()


if __name__ == '__main__':
    main()
