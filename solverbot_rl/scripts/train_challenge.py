#!/usr/bin/env python3
"""Entrenamiento de la política de ALTO NIVEL (selección de habilidades) de un desafío con PPO (MultiDiscrete).

Diseño jerárquico (docs/DESAFIOS_SPEC.md): la política aprende QUÉ habilidad ejecutar y con qué argumento; las
habilidades (navegación, mirada, brazo) son controladores de referencia. Curriculum por tasa de éxito real.

    python3 solverbot_rl/scripts/train_challenge.py --challenge hri --steps 50000 --run hri_ppo
    # pre-entrenamiento opcional por imitación del experto guionado (queda registrado en el resumen)
    python3 solverbot_rl/scripts/train_challenge.py --challenge hri --bc-episodes 40 --steps 50000 --run hri_bc_ppo
    # Gazebo (sim lanzada con world:=home_challenges en el mismo ROS_DOMAIN_ID / GZ_PARTITION; ver tools/headless/)
    python3 solverbot_rl/scripts/train_challenge.py --challenge hri --backend gazebo --steps 5000 --run hri_gz

Reanudación automática desde runs/<run>/checkpoints/latest.zip (modelo, contador, nivel, historial del curriculum);
--fresh lo impide. SIGINT/SIGTERM guardan un checkpoint atómico antes de salir. PPO es on-policy: no hay replay
buffer que persistir (el rollout en curso se descarta al parar). Evaluación independiente periódica con semillas
fijas en un entorno separado (solo backend cinemático; en Gazebo usar evaluate_challenge.py con la sim libre).

Salida en runs/<run>/: config.yaml, episodes.jsonl, progress.jsonl, curriculum.jsonl, evals.jsonl, failures.jsonl,
checkpoints/{latest.zip, latest_state.json, ppo_<pasos>_steps.zip}, model_final.zip, train_summary.json.
"""
import argparse
import json
import os
import signal
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solverbot_rl.challenges.registry import CHALLENGES, levels, make_env, make_expert  # noqa: E402
from solverbot_rl.curriculum import Curriculum, EpisodeLogger  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_challenge import run_episodes  # noqa: E402
from train import atomic_write_json, save_checkpoint  # noqa: E402  (mismos helpers que Pick and Place, sin cambios)

STOP = {'requested': False}


def _on_signal(signum, _frame):
    STOP['requested'] = True
    print(f'[train] señal {signum}: se guardará un checkpoint y se terminará', flush=True)


def behaviour_cloning(model, env, challenge, episodes, seed0, epochs=30, log=print):
    """Imitación supervisada del experto guionado (máxima verosimilitud de sus acciones). Solo inicializa la política."""
    import torch
    obs_l, act_l, succ = [], [], 0
    for i in range(episodes):
        obs, _ = env.reset(seed=seed0 + i)
        ex = make_expert(challenge, env)
        done = False
        while not done:
            a = np.asarray(ex.act(obs))
            obs_l.append(obs)
            act_l.append(a)
            obs, _, te, tr, info = env.step(a)
            done = te or tr
        succ += int(info['episode_summary']['success'])
    log(f'[bc] {len(obs_l)} pares (obs, acción) de {episodes} episodios del experto (éxito {succ}/{episodes})')
    pol = model.policy
    o = torch.as_tensor(np.array(obs_l), dtype=torch.float32, device=pol.device)
    a = torch.as_tensor(np.array(act_l), dtype=torch.long, device=pol.device)
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(o))
        tot = 0.0
        for k in range(0, len(o), 256):
            idx = perm[k:k + 256]
            _, logp, _ = pol.evaluate_actions(o[idx], a[idx])
            loss = -logp.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if ep % 10 == 0 or ep == epochs - 1:
            log(f'[bc] época {ep}: NLL {tot / len(o):.4f}')
    return {'bc_pairs': len(obs_l), 'bc_episodes': episodes, 'bc_expert_success': succ}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--challenge', choices=sorted(CHALLENGES), required=True)
    ap.add_argument('--backend', choices=['kinematic', 'gazebo'], default='kinematic')
    ap.add_argument('--steps', type=int, default=50000, help='pasos (habilidades) TOTALES de la corrida')
    ap.add_argument('--level', type=int, default=0)
    ap.add_argument('--no-curriculum', action='store_true')
    ap.add_argument('--window', type=int, default=20)
    ap.add_argument('--promote-at', type=float, default=0.8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--run', default=None)
    ap.add_argument('--checkpoint-every', type=int, default=2048)
    ap.add_argument('--eval-every', type=int, default=4096, help='0 desactiva (en Gazebo siempre desactivada)')
    ap.add_argument('--eval-episodes', type=int, default=10)
    ap.add_argument('--bc-episodes', type=int, default=0, help='episodios del experto para pre-entrenar (0 = no)')
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cpu')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--fresh', action='store_true')
    args = ap.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import set_random_seed

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    run = args.run or time.strftime(f'{args.challenge}_ppo_{args.backend}_%Y%m%d_%H%M%S')
    out = os.path.join('runs', run)
    ckdir = os.path.join(out, 'checkpoints')
    os.makedirs(out, exist_ok=True)
    set_random_seed(args.seed)
    torch.set_num_threads(args.threads or max(1, (os.cpu_count() or 2) // 2))
    latest = os.path.join(ckdir, 'latest.zip')
    resuming = os.path.exists(latest) and not args.fresh
    state = {}
    if resuming and os.path.exists(os.path.join(ckdir, 'latest_state.json')):
        with open(os.path.join(ckdir, 'latest_state.json')) as fh:
            state = json.load(fh)
    lv = levels(args.challenge)
    env = make_env(args.challenge, backend=args.backend, level=state.get('level', args.level))
    with open(os.path.join(out, 'config.yaml' if not resuming else f'config_resume_{int(time.time())}.yaml'), 'w') as fh:
        yaml.safe_dump({'args': vars(args), 'resumed_from': state or None, 'levels': lv,
                        'hierarchical': True, 'observation_is_privileged': False}, fh, sort_keys=False)
    env = EpisodeLogger(env, os.path.join(out, 'episodes.jsonl'), run)
    env.reset(seed=args.seed + int(state.get('num_timesteps', 0)))
    cur = Curriculum(len(lv), args.window, args.promote_at, state.get('level', args.level))
    for h in state.get('hist', []):
        cur.hist.append(h)
    eval_env = make_env(args.challenge, backend='kinematic') if args.backend == 'kinematic' and args.eval_every else None

    class Monitor(BaseCallback):
        def _on_training_start(self):
            self.t_last, self.n_last = time.time(), self.num_timesteps
            self.next_ck = (self.num_timesteps // args.checkpoint_every + 1) * args.checkpoint_every
            self.next_ev = (self.num_timesteps // args.eval_every + 1) * args.eval_every if eval_env else None

        def _on_step(self) -> bool:
            for info in self.locals['infos']:
                summ = info.get('episode_summary')
                if summ is None:
                    continue
                if not summ['success']:
                    with open(os.path.join(out, 'failures.jsonl'), 'a') as fh:
                        fh.write(json.dumps({'timestep': self.num_timesteps, 'last_skill': info['skill'],
                                             'reason': info['skill_reason'], **summ}, default=str) + '\n')
                if not args.no_curriculum and cur.record(summ['success']):
                    env.unwrapped.set_level(cur.level)
                    rec = {'timestep': self.num_timesteps, 'new_level': cur.level, 'name': lv[cur.level]['name']}
                    with open(os.path.join(out, 'curriculum.jsonl'), 'a') as fh:
                        fh.write(json.dumps(rec) + '\n')
                    print('[curriculum]', rec, flush=True)
                self.logger.record('task/success_rate_window', cur.success_rate)
                self.logger.record('task/level', cur.level)
            if self.num_timesteps >= self.next_ck:
                now = time.time()
                rec = {'timestep': self.num_timesteps, 'level': cur.level, 'time': now,
                       'steps_per_s': round((self.num_timesteps - self.n_last) / max(1e-6, now - self.t_last), 3),
                       'success_rate_window': round(cur.success_rate, 3)}
                with open(os.path.join(out, 'progress.jsonl'), 'a') as fh:
                    fh.write(json.dumps(rec) + '\n')
                save_checkpoint(self.model, ckdir, cur, 'ppo')
                print('[checkpoint]', rec, flush=True)
                self.t_last, self.n_last = now, self.num_timesteps
                self.next_ck += args.checkpoint_every
            if self.next_ev is not None and self.num_timesteps >= self.next_ev:
                eval_env.set_level(cur.level)
                pol = lambda e, s: (lambda o: self.model.predict(o, deterministic=True)[0])  # noqa: E731
                summ, _ = run_episodes(eval_env, pol, args.eval_episodes, 90_000, log=lambda *_: None)
                rec = {'timestep': self.num_timesteps, 'level': cur.level, **summ}
                with open(os.path.join(out, 'evals.jsonl'), 'a') as fh:
                    fh.write(json.dumps(rec) + '\n')
                print('[eval]', rec, flush=True)
                self.next_ev += args.eval_every
            return not STOP['requested']

    bc = None
    if resuming:
        model = PPO.load(latest, env=env, device=args.device)
        print(f'[train] reanudando desde {latest} en t={model.num_timesteps}, nivel {cur.level}', flush=True)
    else:
        model = PPO('MlpPolicy', env, seed=args.seed, verbose=1, device=args.device, n_steps=512, batch_size=128,
                    learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
                    policy_kwargs={'net_arch': [256, 256]})
        if args.bc_episodes:
            bc = behaviour_cloning(model, env.unwrapped, args.challenge, args.bc_episodes, 50_000)
    remaining = max(0, args.steps - int(model.num_timesteps))
    t0, status = time.time(), 'completed'
    try:
        if remaining:
            model.learn(total_timesteps=remaining, callback=[Monitor()], progress_bar=False,
                        reset_num_timesteps=not resuming)
        if STOP['requested']:
            status = 'stopped'
    except Exception as exc:
        status = f'error: {exc!r}'
        raise
    finally:
        save_checkpoint(model, ckdir, cur, 'ppo', keep_numbered=False)
        if status == 'completed':
            model.save(os.path.join(out, 'model_final'))
        atomic_write_json(os.path.join(out, 'train_summary.json'),
                          {'challenge': args.challenge, 'backend': args.backend, 'status': status,
                           'num_timesteps': int(model.num_timesteps), 'target_steps': args.steps,
                           'wall_s_this_session': round(time.time() - t0, 1), 'final_level': cur.level,
                           'success_rate_window': cur.success_rate, 'bc_pretraining': bc,
                           'note': 'política de alto nivel aprendida (PPO); las habilidades son controladores de '
                                   'referencia. El experto guionado NO es esta política.'})
        print(f'[train] {status}: t={model.num_timesteps} nivel={cur.level} éxito(ventana)={cur.success_rate:.3f}',
              flush=True)
        env.close()
        if eval_env:
            eval_env.close()


if __name__ == '__main__':
    main()
