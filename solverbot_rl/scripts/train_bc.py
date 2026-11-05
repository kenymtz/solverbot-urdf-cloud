#!/usr/bin/env python3
"""Behavior Cloning del experto sobre el actor de un modelo SB3 SAC (sin simulador).

    python3 solverbot_rl/scripts/train_bc.py --data runs/reach_bc_v1/demos/expert_var_m runs/reach_bc_v1/demos/expert_var_m_b \
        --out runs/reach_bc_v1/bc_v1

Salida (<out>/, nunca sobrescribe: si existe model.zip aborta salvo --force-new-dir):
    model.zip          modelo SAC completo (actor entrenado por BC, crítico sin entrenar); se carga con
                       solverbot_rl.imitation.bc.load_policy(path) o SAC.load(path)
    bc_summary.json    datos usados, separación por episodios, normalización, métricas train/val, verificación de carga
    bc_curve.jsonl     curva de entrenamiento por época
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solverbot_rl.imitation import bc  # noqa: E402
from solverbot_rl.imitation.dataset import load_datasets, split_by_episode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', required=True, help='directorios de demostraciones / DAgger')
    ap.add_argument('--out', required=True)
    ap.add_argument('--include-failed', action='store_true', help='usar también demostraciones fallidas del experto')
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=600)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-5)
    ap.add_argument('--w-nll', type=float, default=0.1)
    ap.add_argument('--patience', type=int, default=80)
    ap.add_argument('--hide', default='prev_action', help='componentes ocultas al actor (coma); "" = ninguna')
    ap.add_argument('--net-arch', default='256,256')
    ap.add_argument('--weight-near', type=float, default=0.0,
                    help='peso extra para muestras cerca del objetivo: w = 1 + k * exp(-|obj_rel|/0.05)')
    args = ap.parse_args()

    if os.path.exists(os.path.join(args.out, 'model.zip')):
        raise SystemExit(f'{args.out}/model.zip ya existe: no se sobrescriben modelos (usar otro --out)')
    os.makedirs(args.out, exist_ok=True)
    import torch
    torch.set_num_threads(4)

    ds = load_datasets(args.data, only_success=not args.include_failed)
    tr, va = split_by_episode(ds, args.val_frac, args.seed)
    mean, std = bc.compute_normalization(ds.obs[tr])
    hide = tuple(s for s in args.hide.split(',') if s)
    mask = bc.default_actor_mask(hide) if hide else None
    model = bc.make_sac(None, mean, std, actor_mask=mask, seed=args.seed,
                        net_arch=tuple(int(x) for x in args.net_arch.split(',')))
    weights = None
    if args.weight_near > 0:
        d = np.linalg.norm(ds.obs[:, 13:16], axis=1)
        weights = 1.0 + args.weight_near * np.exp(-d / 0.05)
    print(f'[bc] {len(ds.episodes)} episodios, {len(ds)} muestras; train {len(tr)} / val {len(va)} '
          f'(episodios val: {len(set(ds.episode[va].tolist()))}); actor oculta {hide or "nada"}', flush=True)
    res = bc.train_bc(model, ds.obs, ds.actions, tr, va, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                      weight_decay=args.weight_decay, w_nll=args.w_nll, patience=args.patience, seed=args.seed,
                      log_path=os.path.join(args.out, 'bc_curve.jsonl'), sample_weights=weights)
    path = os.path.join(args.out, 'model.zip')
    model.save(path)

    # Verificación: el .zip cargado reproduce exactamente las acciones del actor entrenado
    act = bc.load_policy(path)
    probe = ds.obs[va if len(va) else tr][:256]
    a_mem = model.predict(probe, deterministic=True)[0]
    a_load = np.array([act(o) for o in probe])
    load_diff = float(np.abs(a_mem - a_load).max())
    assert load_diff < 1e-6, load_diff

    zero_baseline = float(np.mean(ds.actions[va] ** 2)) if len(va) else None
    summary = {'data': args.data, 'episodes': len(ds.episodes), 'samples': len(ds),
               'train_samples': int(len(tr)), 'val_samples': int(len(va)),
               'val_episode_seeds': sorted({ds.episodes[i]['seed'] for i in set(ds.episode[va].tolist())}),
               'args': vars(args), 'actor_hidden': list(hide), 'normalization': {'mean': mean.tolist(), 'std': std.tolist()},
               'val_mse_predict_zero': zero_baseline, **res, 'load_check_max_abs_diff': load_diff,
               'model': path, 'policy_inputs': 'obs (31) de SolverBotPickPlaceEnv, object_source=perception'}
    bc.save_json(os.path.join(args.out, 'bc_summary.json'), summary)
    print(f'[bc] mejor época {res["best_epoch"]}: val MSE {res["best_val_mse"]:.4f} '
          f'(predecir 0: {zero_baseline:.4f}); train MSE {res["train"]["mse"]:.4f}; '
          f'carga verificada (|Δa| máx {load_diff:.1e}) -> {path}', flush=True)


if __name__ == '__main__':
    main()
