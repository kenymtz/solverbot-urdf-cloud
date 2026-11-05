#!/usr/bin/env python3
"""Fine-tuning SAC desde la política BC, con evaluación periódica independiente y reanudación.

    python3 solverbot_rl/scripts/finetune_sac.py --backend gazebo --init runs/reach_bc_v1/bc_v1/model.zip \
        --run runs/reach_bc_v1/ft_v1 --level reach_var_m --steps 3000 --eval-every 1000 \
        --eval-seeds 950000-950009 --demos runs/reach_bc_v1/demos/expert_var_m

Procedimiento (ver solverbot_rl/imitation/finetune.py):
  0. Verifica que el actor cargado es idéntico al BC (parámetros y acciones) -> init_check.json.
  1. Evalúa la política ANTES de cualquier actualización (paso 0) con las semillas de evaluación.
  2. Precarga el replay buffer con las demostraciones (transiciones reales de Gazebo).
  3. learning_starts = 0 (el BC actúa desde el primer paso; SB3 usaría acciones uniformes aleatorias si no).
  4. Las primeras --actor-freeze actualizaciones solo entrenan el crítico.
  5. Cada --eval-every pasos: evaluación determinista en semillas fijas no usadas para entrenar + checkpoint.

Reanudación: si <run>/checkpoints/latest.zip existe se continúa (modelo, replay buffer, historial de evaluación).
SIGINT/SIGTERM: termina el paso, guarda checkpoint y sale.
"""
import argparse
import json
import os
import signal
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solverbot_rl.curriculum import EpisodeLogger  # noqa: E402
from solverbot_rl.envs.pick_place import SolverBotPickPlaceEnv  # noqa: E402
from solverbot_rl.imitation import dataset as D  # noqa: E402
from solverbot_rl.imitation.experiment import env_kwargs, level_index, load_experiment, parse_seeds  # noqa: E402
from solverbot_rl.imitation.rollout import run_episode  # noqa: E402

STOP = {'requested': False}


def _on_signal(signum, _frame):
    STOP['requested'] = True
    print(f'[ft] señal {signum}: se guardará un checkpoint y se terminará', flush=True)


def atomic_json(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, path)


def load_transitions(roots):
    """Transiciones completas (incluye episodios fallidos: son experiencia real válida para el crítico)."""
    keys = ('obs', 'next_obs', 'actions', 'rewards', 'terminated', 'truncated')
    out = {k: [] for k in keys}
    for root in roots:
        for row in D.read_index(root):
            with np.load(os.path.join(root, row['file'])) as z:
                if D.validate_episode({k: z[k] for k in z.files}):
                    raise ValueError(f'episodio inválido {root}/{row["file"]}')
                for k in keys:
                    out[k].append(z[k])
    return {k: np.concatenate(v) for k, v in out.items()}


def evaluate(model, env, lvl, seeds, out_dir):
    rows = []
    for s in seeds:
        arrays, summ = run_episode(env, s, lvl, act=lambda o: model.predict(o, deterministic=True)[0])
        summ['expert_in_loop'] = False
        D.save_episode(out_dir, arrays, summ)
        rows.append(summ)
    k = sum(r['success'] for r in rows)
    return {'episodes': len(rows), 'successes': k, 'success_rate': k / len(rows),
            'mean_return': float(np.mean([r['return'] for r in rows])),
            'dmin_mm_mean': float(np.mean([r['dmin_mm'] for r in rows])),
            'dfinal_mm_mean': float(np.mean([r['dfinal_mm'] for r in rows])),
            'length_mean': float(np.mean([r['length'] for r in rows])),
            'collision_rate': float(np.mean([r['collision'] for r in rows])),
            'stalled_rate': float(np.mean([r['stalled'] for r in rows])),
            'failed_seeds': [r['seed'] for r in rows if not r['success']]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', required=True, help='modelo BC (.zip)')
    ap.add_argument('--run', required=True)
    ap.add_argument('--backend', choices=['kinematic', 'gazebo'], default='gazebo')
    ap.add_argument('--experiment', default='reach_bc_v1')
    ap.add_argument('--level', default='reach_var_m')
    ap.add_argument('--steps', type=int, default=3000, help='pasos de entorno TOTALES')
    ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--eval-seeds', default='950000-950009')
    ap.add_argument('--demos', nargs='*', default=[])
    ap.add_argument('--actor-freeze', type=int, default=1000, help='actualizaciones solo-crítico al inicio')
    ap.add_argument('--offline-critic-steps', type=int, default=2000,
                    help='actualizaciones solo-crítico sobre las demostraciones antes de interactuar')
    ap.add_argument('--bc-coef', type=float, default=1.0, help='peso de la regularización BC en el actor')
    ap.add_argument('--q-alpha', type=float, default=2.5, help='TD3+BC: término RL * q_alpha/mean|Q| (0 = SAC sin normalizar)')
    ap.add_argument('--actor-lr-scale', type=float, default=1.0)
    ap.add_argument('--ent-coef', type=float, default=0.01)
    ap.add_argument('--auto-entropy', action='store_true', help='alpha automático (objetivo: entropía del BC)')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train-seed', type=int, default=500000, help='semilla de los episodios de entrenamiento')
    args = ap.parse_args()

    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from solverbot_rl.imitation import bc
    from solverbot_rl.imitation import finetune as FT

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    torch.set_num_threads(4)
    ckdir = os.path.join(args.run, 'checkpoints')
    os.makedirs(ckdir, exist_ok=True)
    latest = os.path.join(ckdir, 'latest.zip')
    state_path = os.path.join(ckdir, 'latest_state.json')
    resuming = os.path.exists(latest)
    log_dir = os.path.join(args.run, 'sb3_log', time.strftime('%Y%m%d_%H%M%S'))   # una carpeta por sesión

    exp = load_experiment(args.experiment)
    base_env = SolverBotPickPlaceEnv(backend=args.backend, **env_kwargs(exp))
    lvl = level_index(base_env, args.level)
    base_env.set_level(lvl)
    env = EpisodeLogger(base_env, os.path.join(args.run, 'train_episodes.jsonl'), os.path.basename(args.run))
    eval_seeds = parse_seeds(args.eval_seeds)

    state = {'evals': [], 'args': vars(args)}
    if resuming:
        with open(state_path) as fh:
            state = json.load(fh)
        model = FT.SACBC.load(latest, env=env, device='cpu')
        model.load_replay_buffer(os.path.join(ckdir, 'latest_replay.pkl'))
        model.actor_freeze_updates, model.bc_coef = state['actor_freeze_abs'], state['args']['bc_coef']
        model.q_alpha, model.actor_lr_scale = state['args']['q_alpha'], state['args']['actor_lr_scale']
        if state['entropy']['auto']:
            model.target_entropy = state['entropy']['target']
        else:
            model.set_entropy(state['entropy']['ent_coef'], learn=False)
        model.set_logger(configure(log_dir, ['csv', 'stdout']))
        print(f'[ft] reanudando en t={model.num_timesteps} (updates={model._n_updates}, '
              f'buffer={model.replay_buffer.size()})', flush=True)
    else:
        if os.path.exists(os.path.join(args.run, 'init_check.json')):
            raise SystemExit(f'{args.run} ya tiene una corrida iniciada sin checkpoint: usar otro --run')
        ref = bc.load_model(args.init)
        model = FT.SACBC.load(args.init, env=env, device='cpu', learning_starts=0, learning_rate=args.lr,
                              batch_size=args.batch_size, seed=args.seed)
        model.set_random_seed(args.train_seed)
        model.actor_freeze_updates, model.bc_coef = args.actor_freeze, args.bc_coef
        model.q_alpha, model.actor_lr_scale = args.q_alpha, args.actor_lr_scale
        # El optimizador del modelo guardado usa la tasa de aprendizaje del BC: se actualiza a --lr
        from stable_baselines3.common.utils import get_schedule_fn
        model.lr_schedule = get_schedule_fn(args.lr)
        # 0. inicialización verificable
        p_ref, p_new = FT.actor_param_vector(ref), FT.actor_param_vector(model)
        probe = None
        demo = load_transitions(args.demos) if args.demos else None
        if demo is not None:
            probe = demo['obs'][:512]
        else:
            probe = np.zeros((1, 31), np.float32)
        da = float(np.abs(ref.predict(probe, deterministic=True)[0] - model.predict(probe, deterministic=True)[0]).max())
        ent_bc = FT.policy_entropy_estimate(model, probe) if demo is not None else None
        model.set_entropy(args.ent_coef, learn=args.auto_entropy, target_entropy=ent_bc)
        check = {'actor_params_equal': bool(np.array_equal(p_ref, p_new)), 'n_actor_params': int(p_ref.size),
                 'max_abs_action_diff': da, 'bc_policy_entropy': ent_bc, 'ent_coef': args.ent_coef,
                 'auto_entropy': args.auto_entropy, 'target_entropy': float(model.target_entropy),
                 'learning_starts': model.learning_starts, 'init': args.init}
        atomic_json(os.path.join(args.run, 'init_check.json'), check)
        print('[ft] verificación de inicialización:', check, flush=True)
        assert check['actor_params_equal'] and da < 1e-6
        state['entropy'] = {'auto': args.auto_entropy, 'ent_coef': args.ent_coef,
                            'target': float(model.target_entropy)}
        model.set_logger(configure(log_dir, ['csv', 'stdout']))
        # 2. demostraciones en el replay buffer + calentamiento offline del crítico
        if demo is not None:
            n = FT.add_demos_to_buffer(model, demo['obs'], demo['next_obs'], demo['actions'], demo['rewards'],
                                       demo['terminated'], demo['truncated'])
            model.set_bc_data(*bc_demo_pairs(args.demos))
            print(f'[ft] {n} transiciones de demostración en el replay buffer', flush=True)
            if args.offline_critic_steps > 0:
                saved = model.actor_freeze_updates
                model.actor_freeze_updates = model._n_updates + args.offline_critic_steps
                p0 = FT.actor_param_vector(model)
                model.train(gradient_steps=args.offline_critic_steps, batch_size=args.batch_size)
                assert np.array_equal(p0, FT.actor_param_vector(model)), 'el actor cambió durante el calentamiento'
                model.actor_freeze_updates = model._n_updates + saved
                print(f'[ft] calentamiento offline del crítico: {args.offline_critic_steps} actualizaciones '
                      f'(actor intacto); el actor se congela hasta la actualización {model.actor_freeze_updates}',
                      flush=True)
        # 1. evaluación antes de actualizar el actor
        ev = evaluate(model, base_env, lvl, eval_seeds, os.path.join(args.run, 'eval', 'step_0'))
        state['evals'].append({'timestep': 0, 'n_updates': model._n_updates, **ev})
        print('[ft] evaluación paso 0 (política BC):', ev, flush=True)
        state['actor_freeze_abs'] = int(model.actor_freeze_updates)
        model._last_obs = None

    if model.bc_coef > 0 and model._bc_obs is None and args.demos:
        model.set_bc_data(*bc_demo_pairs(args.demos))

    class Stopper(BaseCallback):
        def _on_step(self):
            return not STOP['requested']

    def save():
        tmp = os.path.join(ckdir, 'latest_tmp')
        model.save(tmp)
        os.replace(tmp + '.zip', latest)
        model.save_replay_buffer(os.path.join(ckdir, 'latest_replay_tmp'))
        os.replace(os.path.join(ckdir, 'latest_replay_tmp.pkl'), os.path.join(ckdir, 'latest_replay.pkl'))
        state['num_timesteps'], state['n_updates'], state['time'] = int(model.num_timesteps), model._n_updates, time.time()
        atomic_json(state_path, state)

    status = 'completed'
    t0 = time.time()
    try:
        while model.num_timesteps < args.steps and not STOP['requested']:
            chunk = min(args.eval_every - model.num_timesteps % args.eval_every, args.steps - model.num_timesteps)
            p_before = FT.actor_param_vector(model)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False, callback=Stopper(), progress_bar=False)
            moved = float(np.abs(FT.actor_param_vector(model) - p_before).max())
            if STOP['requested']:
                status = 'stopped'
                break
            ev = evaluate(model, base_env, lvl, eval_seeds,
                          os.path.join(args.run, 'eval', f'step_{model.num_timesteps}'))
            rec = {'timestep': int(model.num_timesteps), 'n_updates': model._n_updates,
                   'actor_param_change_max': moved, 'wall_s': round(time.time() - t0, 1), **ev}
            state['evals'].append(rec)
            print('[ft] evaluación:', rec, flush=True)
            model._last_obs = None              # el episodio de entrenamiento se reinicia tras evaluar
            save()
            model.save(os.path.join(ckdir, f'model_{model.num_timesteps}_steps'))
    except Exception as exc:
        status = f'error: {exc!r}'
        raise
    finally:
        save()
        best = max(state['evals'], key=lambda e: (e['success_rate'], -e['dfinal_mm_mean']))
        atomic_json(os.path.join(args.run, 'ft_summary.json'),
                    {'status': status, 'num_timesteps': int(model.num_timesteps), 'evals': state['evals'],
                     'best_eval': best, 'eval_seeds': args.eval_seeds, 'args': vars(args)})
        print(f'[ft] {status}: t={model.num_timesteps}; mejor evaluación en t={best["timestep"]} '
              f'({best["successes"]}/{best["episodes"]})', flush=True)
        base_env.close()


def bc_demo_pairs(roots):
    ds = D.load_datasets(roots, only_success=True)
    return ds.obs, ds.actions


if __name__ == '__main__':
    main()
