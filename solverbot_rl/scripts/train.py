#!/usr/bin/env python3
"""Entrenamiento con Stable-Baselines3 (SAC por defecto, PPO opcional), curriculum por tasa de éxito y ejecución
autónoma: checkpoints atómicos, reanudación automática y parada limpia (SIGINT/SIGTERM guarda antes de salir).

    # Sin simulador (backend cinemático, SIN física; valida el pipeline):
    python3 solverbot_rl/scripts/train.py --backend kinematic --algo ppo --steps 60000 --run ppo_kin

    # Gazebo (simulación en marcha en el mismo ROS_DOMAIN_ID / GZ_PARTITION; ver tools/headless/):
    python3 solverbot_rl/scripts/train.py --backend gazebo --algo sac --steps 200000 --run sac_gz

Si runs/<run>/checkpoints/latest.zip existe, la corrida se REANUDA desde allí (modelo, replay buffer, contador de
pasos, nivel del curriculum). --fresh lo impide.

Salida en runs/<run>/: config.yaml, episodes.jsonl, progress.jsonl (rendimiento), curriculum.jsonl,
checkpoints/{latest.zip, latest_replay.pkl, latest_state.json, <algo>_<pasos>.zip}, model_final.zip, train_summary.json.
"""
import argparse
import json
import os
import signal
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solverbot_rl.curriculum import Curriculum, EpisodeLogger  # noqa: E402
from solverbot_rl.envs.pick_place import SolverBotPickPlaceEnv  # noqa: E402

STOP = {'requested': False}


def _on_signal(signum, _frame):
    STOP['requested'] = True
    print(f'[train] señal {signum}: se guardará un checkpoint y se terminará', flush=True)


def atomic_write_json(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, path)


def save_checkpoint(model, ckdir, cur, algo, keep_numbered=True):
    """Guarda modelo (+ replay buffer en SAC) y estado de forma atómica: escribe a .tmp y renombra."""
    os.makedirs(ckdir, exist_ok=True)
    tmp = os.path.join(ckdir, 'latest_tmp')
    model.save(tmp)
    os.replace(tmp + '.zip', os.path.join(ckdir, 'latest.zip'))
    if hasattr(model, 'save_replay_buffer'):
        model.save_replay_buffer(os.path.join(ckdir, 'latest_replay_tmp'))
        os.replace(os.path.join(ckdir, 'latest_replay_tmp.pkl'), os.path.join(ckdir, 'latest_replay.pkl'))
    atomic_write_json(os.path.join(ckdir, 'latest_state.json'),
                      {'num_timesteps': int(model.num_timesteps), 'level': cur.level, 'hist': list(cur.hist),
                       'time': time.time()})
    if keep_numbered:
        model.save(os.path.join(ckdir, f'{algo}_{model.num_timesteps}_steps'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['kinematic', 'gazebo'], default='kinematic')
    ap.add_argument('--algo', choices=['sac', 'ppo'], default='sac')
    ap.add_argument('--steps', type=int, default=20000, help='pasos TOTALES de la corrida (incluye los ya hechos)')
    ap.add_argument('--level', type=int, default=0, help='nivel inicial del curriculum (corrida nueva)')
    ap.add_argument('--no-curriculum', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--run', default=None)
    ap.add_argument('--config', default=None)
    ap.add_argument('--object-source', choices=['perception', 'privileged', 'none'], default=None)
    ap.add_argument('--images', default=None, help='lista separada por comas, p. ej. wrist (solo gazebo)')
    ap.add_argument('--checkpoint-every', type=int, default=5000)
    ap.add_argument('--fresh', action='store_true', help='no reanudar aunque exista latest.zip')
    ap.add_argument('--resume', default=None, help='(compatibilidad) ruta explícita a un .zip')
    args = ap.parse_args()

    import torch
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import set_random_seed

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    run = args.run or time.strftime(f'{args.algo}_{args.backend}_%Y%m%d_%H%M%S')
    out = os.path.join('runs', run)
    ckdir = os.path.join(out, 'checkpoints')
    os.makedirs(out, exist_ok=True)
    overrides = {}
    if args.object_source:
        overrides['observation.object_source'] = args.object_source
    if args.images is not None:
        overrides['observation.images'] = [s for s in args.images.split(',') if s]
    set_random_seed(args.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

    latest = os.path.join(ckdir, 'latest.zip')
    resuming = (os.path.exists(latest) and not args.fresh) or args.resume
    state = {}
    if resuming and os.path.exists(os.path.join(ckdir, 'latest_state.json')):
        with open(os.path.join(ckdir, 'latest_state.json')) as fh:
            state = json.load(fh)

    env = SolverBotPickPlaceEnv(backend=args.backend, config_path=args.config,
                                level=state.get('level', args.level), overrides=overrides)
    cfg_path = os.path.join(out, 'config.yaml' if not resuming else f'config_resume_{int(time.time())}.yaml')
    with open(cfg_path, 'w') as fh:
        yaml.safe_dump({'args': vars(args), 'resumed_from': state or None,
                        'observation_is_privileged': env.object_source == 'privileged', 'task': env.cfg},
                       fh, sort_keys=False)
    if env.object_source == 'privileged':
        print('[train] AVISO: object_source=privileged (pose real del simulador en la observación). '
              'Una política así NO se declara basada en visión.', flush=True)
    env = EpisodeLogger(env, os.path.join(out, 'episodes.jsonl'), run)
    env.reset(seed=args.seed + int(state.get('num_timesteps', 0)))
    cc = env.unwrapped.cfg['curriculum']
    cur = Curriculum(len(cc['levels']), cc['window'], cc['promote_at'], state.get('level', args.level))
    for h in state.get('hist', []):
        cur.hist.append(h)

    class Monitor(BaseCallback):
        """Curriculum por tasa de éxito, checkpoints atómicos periódicos, métricas de rendimiento y parada limpia."""

        def _on_training_start(self):
            self.t_last, self.n_last, self.next_ck = time.time(), self.num_timesteps, \
                (self.num_timesteps // args.checkpoint_every + 1) * args.checkpoint_every

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
            if self.num_timesteps >= self.next_ck:
                now = time.time()
                sps = (self.num_timesteps - self.n_last) / max(1e-6, now - self.t_last)
                rec = {'timestep': self.num_timesteps, 'steps_per_s': round(sps, 3), 'level': cur.level,
                       'success_rate_window': round(cur.success_rate, 3), 'time': now}
                with open(os.path.join(out, 'progress.jsonl'), 'a') as fh:
                    fh.write(json.dumps(rec) + '\n')
                save_checkpoint(self.model, ckdir, cur, args.algo)
                print('[checkpoint]', rec, flush=True)
                self.t_last, self.n_last = now, self.num_timesteps
                self.next_ck += args.checkpoint_every
            return not STOP['requested']

    policy = 'MultiInputPolicy' if hasattr(env.observation_space, 'spaces') else 'MlpPolicy'
    algo = SAC if args.algo == 'sac' else PPO
    if resuming:
        path = args.resume or latest
        model = algo.load(path, env=env)
        rb = os.path.join(ckdir, 'latest_replay.pkl')
        if hasattr(model, 'load_replay_buffer') and os.path.exists(rb) and not args.resume:
            model.load_replay_buffer(rb)
        print(f'[train] reanudando desde {path} en t={model.num_timesteps}, nivel {cur.level}', flush=True)
    else:
        kw = dict(seed=args.seed, verbose=1)
        if args.algo == 'sac':
            kw.update(learning_starts=min(2000, args.steps // 5), buffer_size=200_000, batch_size=256, train_freq=1,
                      gradient_steps=1, learning_rate=3e-4)
            if policy == 'MultiInputPolicy':
                kw['buffer_size'] = 50_000           # imágenes: memoria
        else:
            kw.update(n_steps=1024, batch_size=256, learning_rate=3e-4, gae_lambda=0.95, gamma=0.99, ent_coef=0.0)
        model = algo(policy, env, **kw)
    remaining = max(0, args.steps - int(model.num_timesteps))
    t0 = time.time()
    status = 'completed'
    try:
        if remaining:
            model.learn(total_timesteps=remaining, callback=[Monitor()], progress_bar=False,
                        reset_num_timesteps=not resuming)
        if STOP['requested']:
            status = 'stopped'
    except Exception as exc:                     # se guarda lo aprendido antes de propagar
        status = f'error: {exc!r}'
        raise
    finally:
        save_checkpoint(model, ckdir, cur, args.algo, keep_numbered=False)
        if status == 'completed':
            model.save(os.path.join(out, 'model_final'))
        atomic_write_json(os.path.join(out, 'train_summary.json'),
                          {'status': status, 'num_timesteps': int(model.num_timesteps), 'target_steps': args.steps,
                           'wall_s_this_session': round(time.time() - t0, 1), 'final_level': cur.level,
                           'success_rate_window': cur.success_rate, 'object_source': env.unwrapped.object_source})
        print(f'[train] {status}: t={model.num_timesteps} nivel={cur.level} éxito(ventana)={cur.success_rate:.3f}',
              flush=True)
        env.close()


if __name__ == '__main__':
    main()
