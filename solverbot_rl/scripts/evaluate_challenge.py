#!/usr/bin/env python3
"""Evaluación reproducible de un desafío (HRI / GPSR / Laundry): N episodios con semillas fijas (seed0 + i).
Mide la TASA DE ÉXITO real (criterio verificable con la verdad del simulador) y registra cada fallo con su traza.

    # experto guionado de referencia (cota superior; NO es una política aprendida)
    python3 solverbot_rl/scripts/evaluate_challenge.py --challenge hri --policy expert --level 5 --episodes 20
    # línea base aleatoria
    python3 solverbot_rl/scripts/evaluate_challenge.py --challenge hri --policy random --level 0 --episodes 20
    # política entrenada (PPO)
    python3 solverbot_rl/scripts/evaluate_challenge.py --challenge hri --policy runs/<run>/model_final.zip --level 2
    # Gazebo (simulación lanzada con world:=home_challenges en el mismo ROS_DOMAIN_ID / GZ_PARTITION)
    python3 solverbot_rl/scripts/evaluate_challenge.py --challenge hri --backend gazebo --policy expert --level 3

Salida: JSON (--out) con resumen + un registro por episodio (semilla, éxito, progreso, tiempo simulado, colisión,
error de localización, últimos pasos y eventos si falló).
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solverbot_rl.challenges.registry import CHALLENGES, make_env, make_expert  # noqa: E402


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def run_episodes(env, policy, episodes, seed0, max_steps=400, log=print, sink=None):
    rows = [] if sink is None else sink
    for i in range(episodes):
        seed = seed0 + i
        obs, info = env.reset(seed=seed)
        act = policy(env, seed)
        trace, done, ret, t0 = [], False, 0.0, time.time()
        steps = 0
        while not done and steps < max_steps:
            a = act(obs)
            obs, r, te, tr, info = env.step(a)
            ret += r
            steps += 1
            done = te or tr
            trace.append([info['skill'], info['arg'], info['skill_ok'], info['skill_reason']])
        summ = info.get('episode_summary', {'success': False, 'progress': info.get('progress')})
        row = {'seed': seed, 'success': bool(summ['success']), 'return': round(ret, 3), 'skills': steps,
               'wall_s': round(time.time() - t0, 1), **{k: v for k, v in summ.items() if k != 'success'}}
        if not row['success']:
            row['last_steps'] = trace[-10:]
            row['last_events'] = getattr(env.unwrapped, 'events', [])[-5:]
        rows.append(_jsonable(row))
        log(f"[eval] seed {seed}: {'ÉXITO' if row['success'] else 'falla'} ret={ret:.2f} pasos={steps} "
            f"t_sim={summ.get('sim_time')} colisión={summ.get('contact')}")
    succ = np.array([r['success'] for r in rows], float)
    n = len(rows)
    return {'episodes': n, 'successes': int(succ.sum()), 'success_rate': float(succ.mean()) if n else 0.0,
            'success_ci95': float(1.96 * np.sqrt(succ.mean() * (1 - succ.mean()) / n)) if n else 0.0,
            'collisions': int(sum(bool(r.get('contact')) for r in rows)),
            'tipped': int(sum(bool(r.get('tipped')) for r in rows)),
            'mean_return': float(np.mean([r['return'] for r in rows])) if n else 0.0,
            'mean_sim_time': float(np.mean([r.get('sim_time', 0) for r in rows])) if n else 0.0,
            'loc_err_mean': float(np.mean([r.get('loc_err_mean', 0) for r in rows])) if n else 0.0,
            'loc_err_max': float(np.max([r.get('loc_err_max', 0) for r in rows])) if n else 0.0}, rows


def make_policy(spec, challenge):
    if spec == 'expert':
        def p(env, seed):
            ex = make_expert(challenge, env)
            return ex.act
        return p
    if spec == 'random':
        def p(env, seed):
            rng = np.random.default_rng(seed)
            return lambda obs: rng.integers(env.action_space.nvec)
        return p
    from stable_baselines3 import PPO
    model = PPO.load(spec, device='cpu')

    def p(env, seed):
        return lambda obs: model.predict(obs, deterministic=True)[0]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--challenge', choices=sorted(CHALLENGES), required=True)
    ap.add_argument('--backend', choices=['kinematic', 'gazebo'], default='kinematic')
    ap.add_argument('--policy', default='expert', help='expert | random | ruta a un .zip de PPO')
    ap.add_argument('--level', type=int, default=0)
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--seed0', type=int, default=1000)
    ap.add_argument('--nav-pose', choices=['estimate', 'truth'], default='estimate')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    env = make_env(args.challenge, backend=args.backend, level=args.level, nav_pose_source=args.nav_pose)
    t0 = time.time()
    error = None
    rows_partial = []
    try:
        summary, rows = run_episodes(env, make_policy(args.policy, args.challenge), args.episodes, args.seed0,
                                     sink=rows_partial)
    except Exception as exc:                    # p. ej. simulación que deja de avanzar: se guarda lo medido
        error = repr(exc)
        rows = rows_partial
        summary = {'episodes': len(rows), 'successes': sum(r['success'] for r in rows), 'aborted': True}
    finally:
        env.close()
    rep = {'challenge': args.challenge, 'backend': args.backend, 'physical': args.backend == 'gazebo',
           'policy': args.policy, 'policy_is_learned': args.policy not in ('expert', 'random'),
           'level': args.level, 'seed0': args.seed0, 'nav_pose_source': args.nav_pose,
           'wall_s': round(time.time() - t0, 1), 'host': platform.node(),
           'ros_domain_id': os.environ.get('ROS_DOMAIN_ID'), 'gz_partition': os.environ.get('GZ_PARTITION'),
           'error': error, 'summary': summary, 'episodes': rows}
    print(json.dumps({k: v for k, v in rep.items() if k != 'episodes'}, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out + '.tmp', 'w') as fh:
            json.dump(rep, fh, indent=1)
        os.replace(args.out + '.tmp', args.out)
    if error:
        sys.exit(f'[eval] abortado: {error}')


if __name__ == '__main__':
    main()
