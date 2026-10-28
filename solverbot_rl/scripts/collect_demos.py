#!/usr/bin/env python3
"""Recolección de demostraciones del experto guionado (o DAgger: estados de una política, etiquetas del experto).

    # Demostraciones del experto en Gazebo (simulación headless en marcha en la instancia; ver tools/headless):
    python3 solverbot_rl/scripts/collect_demos.py --backend gazebo --experiment reach_bc_v1 --level reach_var_m \
        --seeds 100000-100059 --out runs/reach_bc_v1/demos/expert_var_m

    # DAgger: ejecuta la política (beta=0) y registra la acción del experto como etiqueta en cada estado visitado
    python3 solverbot_rl/scripts/collect_demos.py --backend gazebo --experiment reach_bc_v1 --level reach_var_m \
        --seeds 110000-110029 --policy runs/reach_bc_v1/bc_v1/model.zip --beta 0 --out runs/reach_bc_v1/demos/dagger1

Reanudable: las semillas ya presentes en <out>/index.jsonl se saltean. Cada episodio se valida (NaN, dimensiones,
continuidad obs/next_obs) y se registra su resultado físico (éxito con el criterio de task.success, distancias,
colisión, falta de progreso, tasa de detección y error de la percepción respecto de la verdad del simulador).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solverbot_rl.envs.pick_place import SolverBotPickPlaceEnv  # noqa: E402
from solverbot_rl.expert import ScriptedExpert  # noqa: E402
from solverbot_rl.imitation import dataset as D  # noqa: E402
from solverbot_rl.imitation.experiment import env_kwargs, level_index, load_experiment, parse_seeds  # noqa: E402
from solverbot_rl.imitation.rollout import run_episode  # noqa: E402


def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except Exception:  # pragma: no cover
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['kinematic', 'gazebo'], default='gazebo')
    ap.add_argument('--experiment', default='reach_bc_v1')
    ap.add_argument('--level', default='reach')
    ap.add_argument('--seeds', required=True, help='p. ej. 100000-100049')
    ap.add_argument('--out', required=True)
    ap.add_argument('--policy', default=None, help='modelo SB3 (.zip) para DAgger; sin él, el experto ejecuta')
    ap.add_argument('--beta', type=float, default=0.0, help='DAgger: prob. de ejecutar la acción del experto')
    ap.add_argument('--source', default=None, help='etiqueta del conjunto (expert | dagger_<n>)')
    ap.add_argument('--labeler', choices=['scripted', 'local'], default='scripted',
                    help='scripted = ScriptedExpert validado; local = ReachLabelExpert (IK local en cada paso)')
    args = ap.parse_args()

    exp = load_experiment(args.experiment)
    env = SolverBotPickPlaceEnv(backend=args.backend, **env_kwargs(exp))
    if env.object_source != 'perception':
        raise SystemExit(f'object_source={env.object_source}: las demostraciones deben usar perception')
    lvl = level_index(env, args.level)
    if args.labeler == 'local':
        from solverbot_rl.imitation.label_expert import ReachLabelExpert
        expert = ReachLabelExpert(env)
    else:
        expert = ScriptedExpert(env)
    act = None
    if args.policy:
        from solverbot_rl.imitation.bc import load_policy
        act = load_policy(args.policy)
    source = args.source or ('expert' if act is None else 'dagger')
    os.makedirs(args.out, exist_ok=True)
    meta_path = os.path.join(args.out, 'meta.json')
    meta = {'experiment': exp['name'], 'level': env.levels[lvl], 'level_index': lvl, 'backend': args.backend,
            'physical': env.backend.physical, 'object_source': env.object_source, 'source': source,
            'policy': args.policy, 'labeler': args.labeler, 'beta': args.beta if act is not None else 1.0, 'commit': git_commit(),
            'obs_dim': int(env.observation_space.shape[0]), 'act_dim': int(env.action_space.shape[0]),
            'control': env.cfg['control'], 'task': env.cfg['task'], 'reward': env.cfg['reward']}
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            old = json.load(fh)
        for k in ('experiment', 'level_index', 'backend', 'source', 'object_source'):
            if old.get(k) != meta[k]:
                raise SystemExit(f'{args.out} ya contiene datos con {k}={old.get(k)!r} != {meta[k]!r}')
    else:
        with open(meta_path, 'w') as fh:
            json.dump(meta, fh, indent=1)

    done = {r['seed'] for r in D.read_index(args.out)}
    seeds = [s for s in parse_seeds(args.seeds) if s not in done]
    print(f'[collect] {len(done)} episodios previos, {len(seeds)} por recolectar en {args.out} '
          f'(nivel {lvl}:{env.levels[lvl]["name"]}, fuente {source}, backend {args.backend})', flush=True)
    t0 = time.time()
    for i, seed in enumerate(seeds):
        arrays, summ = run_episode(env, seed, lvl, act=act, expert=expert, beta=args.beta,
                                   rng=np.random.default_rng(seed))
        errs = D.validate_episode(arrays)
        summ['valid_structure'] = not errs
        summ['structure_errors'] = errs
        summ['source'] = source
        D.save_episode(args.out, arrays, summ)
        print(f'[collect] {i + 1}/{len(seeds)} seed={seed} éxito={summ["success"]} pasos={summ["length"]} '
              f'd0={summ["d0_mm"]:.1f} dmin={summ["dmin_mm"]:.1f} dfin={summ["dfinal_mm"]:.1f} mm '
              f'zf={summ["tcp_z2_final"]:.3f} det={summ["valid_rate"]:.2f} '
              f'perc={summ["perc_err_mean_mm"] if summ["perc_err_mean_mm"] is None else round(summ["perc_err_mean_mm"], 1)} mm '
              f'col={summ["collision"]} estr={"ok" if not errs else errs} ({summ["wall_s"]:.0f} s)', flush=True)
    rows = D.read_index(args.out)
    n_ok = sum(r['success'] for r in rows)
    print(f'[collect] total {len(rows)} episodios, {n_ok} exitosos, {len(rows) - n_ok} fallidos; '
          f'{time.time() - t0:.0f} s', flush=True)
    env.close()


if __name__ == '__main__':
    main()
