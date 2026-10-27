#!/usr/bin/env python3
"""Evaluación reproducible: N episodios con semillas fijas (seed0 + i) y política determinista.
Mide la TASA DE ÉXITO real (criterio verificable de la tarea), no solo la recompensa.

    python3 solverbot_rl/scripts/evaluate.py --policy runs/<run>/model_final.zip --algo ppo --level 0 --episodes 50
    python3 solverbot_rl/scripts/evaluate.py --policy random   --level 0      # línea base
    python3 solverbot_rl/scripts/evaluate.py --policy scripted --level 3      # experto guionado (cota superior)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solverbot_rl.envs.pick_place import SolverBotPickPlaceEnv  # noqa: E402
from solverbot_rl.expert import ScriptedExpert  # noqa: E402


def evaluate(env, act, episodes, seed0, on_reset=None):
    rows = []
    for i in range(episodes):
        obs, info = env.reset(seed=seed0 + i)
        if on_reset:
            on_reset()
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(act(obs))
            done = term or trunc
        rows.append(info['episode_summary'])
    succ = np.array([r['success'] for r in rows], float)
    rets = np.array([r['return'] for r in rows])
    n = len(rows)
    ci = 1.96 * np.sqrt(max(succ.mean() * (1 - succ.mean()), 1e-9) / n)
    return {'episodes': n, 'success_rate': float(succ.mean()), 'success_ci95': float(ci),
            'mean_return': float(rets.mean()), 'std_return': float(rets.std()),
            'mean_length': float(np.mean([r['length'] for r in rows])),
            'grasp_rate': float(np.mean([r['grasped'] for r in rows])),
            'lift_rate': float(np.mean([r['lifted'] for r in rows])),
            'stalled_rate': float(np.mean([r['stalled'] for r in rows])),
            'collision_rate': float(np.mean(['pen_collision' in r['terms'] for r in rows])),
            'drop_rate': float(np.mean(['pen_drop' in r['terms'] for r in rows]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True, help='ruta .zip | random | scripted')
    ap.add_argument('--algo', choices=['sac', 'ppo'], default='sac')
    ap.add_argument('--backend', choices=['kinematic', 'gazebo'], default='kinematic')
    ap.add_argument('--level', type=int, default=0)
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--seed', type=int, default=10_000, help='semilla base (distinta de la de entrenamiento)')
    ap.add_argument('--object-source', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    overrides = {'observation.object_source': args.object_source} if args.object_source else {}
    env = SolverBotPickPlaceEnv(backend=args.backend, level=args.level, overrides=overrides)
    on_reset = None
    if args.policy == 'random':
        rng = np.random.default_rng(args.seed)

        def act(_):
            return rng.uniform(-1, 1, 7).astype(np.float32)
    elif args.policy == 'scripted':
        ex = ScriptedExpert(env)
        on_reset = ex.reset

        def act(_):
            return ex.act()
    else:
        from stable_baselines3 import PPO, SAC
        model = (SAC if args.algo == 'sac' else PPO).load(args.policy)

        def act(obs):
            return model.predict(obs, deterministic=True)[0]
    res = evaluate(env, act, args.episodes, args.seed, on_reset)
    res.update({'policy': args.policy, 'backend': args.backend, 'level': args.level,
                'level_name': env.levels[args.level]['name'], 'object_source': env.object_source,
                'physical': env.backend.physical})
    print(json.dumps(res, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as fh:
            json.dump(res, fh, indent=1)
    env.close()


if __name__ == '__main__':
    main()
