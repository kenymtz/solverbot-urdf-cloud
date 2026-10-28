#!/usr/bin/env python3
"""Tabla comparativa y gráficas a partir de las evaluaciones de eval_reach.py.

    python3 solverbot_rl/scripts/report_reach.py --eval runs/reach_bc_v1/eval --out runs/reach_bc_v1/report

Lee <eval>/<política>/<nivel>/summary.json + index.jsonl + episodes/*.npz. Escribe <out>/table.md, <out>/table.json,
<out>/dist_<nivel>.png (distancia al punto de aproximación por paso, todas las políticas) y, si se indican
--bc-curves, la curva de entrenamiento del BC.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solverbot_rl.imitation import dataset as D  # noqa: E402

LEVELS = ('reach', 'reach_var_s', 'reach_var_m')


def fmt(x, nd=1):
    return '—' if x is None else f'{x:.{nd}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bc-curves', nargs='*', default=[])
    args = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for path in sorted(glob.glob(os.path.join(args.eval, '*', '*', 'summary.json'))):
        with open(path) as fh:
            s = json.load(fh)
        pol, lvl = path.split(os.sep)[-3:-1]
        rows.append({**s, 'policy_path': s['policy'], 'level_index': s['level'], 'policy': pol, 'level': lvl})
    rows.sort(key=lambda r: (LEVELS.index(r['level']) if r['level'] in LEVELS else 9, r['policy']))
    lines = ['| nivel | política | éxitos | tasa (IC95 Wilson) | d0 mm | dmin mm | dfinal mm | Z TCP final | '
             'pasos | retorno | colisión | sin progreso | límite 150 | detección válida |',
             '|' + '---|' * 14]
    for r in rows:
        lo, hi = r['success_wilson95']
        lines.append(f"| {r['level']} | {r['policy']} | {r['successes']}/{r['episodes']} | {r['success_rate']:.0%} "
                     f"[{lo:.0%}, {hi:.0%}] | {fmt(r['d0_mm']['mean'])} | {fmt(r['dmin_mm']['mean'])} | "
                     f"{fmt(r['dfinal_mm']['mean'])} | {fmt(r['tcp_z2_final']['mean'], 3)} | "
                     f"{fmt(r['length']['mean'])} | {fmt(r['return']['mean'], 2)} | {r['collision_rate']:.0%} | "
                     f"{r['stalled_rate']:.0%} | {r['timeout_rate']:.0%} | {r['detection_valid_rate']:.0%} |")
    with open(os.path.join(args.out, 'table.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    with open(os.path.join(args.out, 'table.json'), 'w') as fh:
        json.dump(rows, fh, indent=1)
    print('\n'.join(lines))

    for lvl in sorted({r['level'] for r in rows}):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, r in enumerate([r for r in rows if r['level'] == lvl]):
            root = os.path.join(args.eval, r['policy'], lvl)
            for j, e in enumerate(D.read_index(root)):
                with np.load(os.path.join(root, e['file'])) as z:
                    ax.plot(1000 * z['dist'], color=f'C{i}', alpha=0.35, lw=1,
                            label=f"{r['policy']} ({r['successes']}/{r['episodes']})" if j == 0 else None)
        ax.axhline(30, color='k', ls='--', lw=1, label='umbral de éxito 30 mm')
        ax.set_xlabel('paso (0.1 s)')
        ax.set_ylabel('distancia TCP–punto de aproximación (mm)')
        ax.set_title(f'Gazebo, nivel {lvl}: episodios de evaluación')
        ax.set_ylim(0, 300)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f'dist_{lvl}.png'), dpi=110)
        plt.close(fig)

    if args.bc_curves:
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, c in enumerate(args.bc_curves):
            recs = [json.loads(line) for line in open(os.path.join(c, 'bc_curve.jsonl'))]
            ep = [r['epoch'] for r in recs]
            ax.plot(ep, [r['train_mse'] for r in recs], color=f'C{i}', ls='--', label=f'{os.path.basename(c)} train')
            ax.plot(ep, [r['val_mse'] for r in recs], color=f'C{i}', label=f'{os.path.basename(c)} val')
        ax.set_yscale('log')
        ax.set_xlabel('época')
        ax.set_ylabel('MSE acción')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, 'bc_curves.png'), dpi=110)
        plt.close(fig)


if __name__ == '__main__':
    main()
