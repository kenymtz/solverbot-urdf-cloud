#!/usr/bin/env python3
"""Entrenamiento con Stable-Baselines3 (SAC por defecto, PPO opcional) y curriculum por tasa de éxito.

    # Prueba rápida sin simulador (backend cinemático, SIN física; valida el pipeline):
    python3 solverbot_rl/scripts/train.py --backend kinematic --algo ppo --steps 30000 --run reach_kin

    # Gazebo Harmonic (en WSL; la simulación debe estar corriendo, ver README):
    python3 solverbot_rl/scripts/train.py --backend gazebo --algo sac --steps 200000 --run pp_gz

Salida en runs/<run>/: config.yaml, episodes.jsonl (un registro por episodio), checkpoints/, model_final.zip,
curriculum.jsonl (promociones de nivel con la tasa de éxito que las justificó).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solverbot_rl.curriculum import Curriculum, EpisodeLogger  # noqa: E402
from solverbot_rl.envs.pick_place import SolverBotPickPlaceEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['kinematic', 'gazebo'], default='kinematic')
    ap.add_argument('--algo', choices=['sac', 'ppo'], default='sac')
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--level', type=int, default=0, help='nivel inicial del curriculum')
    ap.add_argument('--no-curriculum', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--run', default=None)
    ap.add_argument('--config', default=None)
    ap.add_argument('--object-source', choices=['perception', 'privileged', 'none'], default=None)
    ap.add_argument('--images', default=None, help='lista separada por comas, p. ej. wrist (solo gazebo)')
    ap.add_argument('--checkpoint-every', type=int, default=10000)
    ap.add_argument('--resume', default=None, help='ruta a un .zip para continuar')
    args = ap.parse_args()

    import torch
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.utils import set_random_seed

    run = args.run or time.strftime(f'{args.algo}_{args.backend}_%Y%m%d_%H%M%S')
    out = os.path.join('runs', run)
    os.makedirs(out, exist_ok=True)
    overrides = {}
    if args.object_source:
        overrides['observation.object_source'] = args.object_source
    if args.images is not None:
        overrides['observation.images'] = [s for s in args.images.split(',') if s]
    set_random_seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    env = SolverBotPickPlaceEnv(backend=args.backend, config_path=args.config, level=args.level, overrides=overrides)
    with open(os.path.join(out, 'config.yaml'), 'w') as fh:
        yaml.safe_dump({'args': vars(args), 'task': env.cfg}, fh, sort_keys=False)
    env = EpisodeLogger(env, os.path.join(out, 'episodes.jsonl'), run)
    env.reset(seed=args.seed)
    cc = env.unwrapped.cfg['curriculum']
    cur = Curriculum(len(cc['levels']), cc['window'], cc['promote_at'], args.level)

    class CurriculumCallback(BaseCallback):
        def _on_step(self) -> bool:
            for info in self.locals['infos']:
                summ = info.get('episode_summary')
                if summ is None:
                    continue
                if not args.no_curriculum and cur.record(summ['success']):
                    env.unwrapped.set_level(cur.level)
                    rec = {'timestep': self.num_timesteps, 'new_level': cur.level,
                           'name': cc['levels'][cur.level]['name'], 'promote_at': cc['promote_at']}
                    with open(os.path.join(out, 'curriculum.jsonl'), 'a') as fh:
                        fh.write(json.dumps(rec) + '\n')
                    print('[curriculum]', rec, flush=True)
                self.logger.record('task/success_rate_window', cur.success_rate)
                self.logger.record('task/level', cur.level)
            return True

    policy = 'MultiInputPolicy' if isinstance(env.observation_space, dict) or hasattr(env.observation_space, 'spaces') \
        else 'MlpPolicy'
    algo = SAC if args.algo == 'sac' else PPO
    kw = dict(seed=args.seed, verbose=1)
    if args.algo == 'sac':
        kw.update(learning_starts=min(2000, args.steps // 5), buffer_size=200_000, batch_size=256, train_freq=1,
                  gradient_steps=1, learning_rate=3e-4)
        if policy == 'MultiInputPolicy':
            kw['buffer_size'] = 50_000           # imágenes: memoria
    else:
        kw.update(n_steps=1024, batch_size=256, learning_rate=3e-4, gae_lambda=0.95, gamma=0.99, ent_coef=0.0)
    model = algo.load(args.resume, env=env) if args.resume else algo(policy, env, **kw)
    cbs = [CurriculumCallback(), CheckpointCallback(args.checkpoint_every, os.path.join(out, 'checkpoints'),
                                                    name_prefix=args.algo)]
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=cbs, progress_bar=False)
    model.save(os.path.join(out, 'model_final'))
    with open(os.path.join(out, 'train_summary.json'), 'w') as fh:
        json.dump({'steps': args.steps, 'wall_s': time.time() - t0, 'final_level': cur.level,
                   'success_rate_window': cur.success_rate}, fh, indent=1)
    print('modelo guardado en', out, '| nivel final', cur.level, '| éxito (ventana)', round(cur.success_rate, 3))
    env.close()


if __name__ == '__main__':
    main()
