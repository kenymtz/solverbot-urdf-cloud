#!/usr/bin/env python3
"""Fase F (offline): rama de muñeca (joint4) frente a la pérdida de detección en trazas de evaluación ya registradas.

    python3 solverbot_rl/scripts/analyze_branch.py --runs runs/reach_bc_v1/eval/bc_v2w runs/reach_bc_v1/eval_confirm/bc_v2w \
        --out runs/dual_camera/branch_analysis

Por episodio: signo inicial de la acción de joint4, cruces de signo SOSTENIDOS de esa acción (de <= -0.5 a >= +0.5 o
al revés), el paso en que empieza cada cambio (último cruce por cero antes de llegar al otro lado), si la detección de
la muñeca era válida en ese paso y cuántos pasos llevaba perdida, inclinación mínima del TCP (z del eje de la pinza)
y resultado. Solo lee los .npz (obs/acciones de la política y métricas privilegiadas ya separadas).
"""
import argparse
import glob
import json
import os

import numpy as np


def switches(a4, hi=0.5):
    """Cambios sostenidos de rama en la acción de joint4: lista de (inicio, fin, dirección)."""
    out, side, side_start = [], 0, 0
    for t, v in enumerate(a4):
        s = 1 if v >= hi else (-1 if v <= -hi else 0)
        if s == 0:
            continue
        if side and s != side:
            # el cambio "empieza" en el paso siguiente al último en que la acción estaba en la rama anterior con
            # |a4| >= 0.9 (o con su máximo, si nunca saturó) y cruza 0 en el primer cambio de signo posterior
            seg = side * a4[side_start:t]
            sat = np.nonzero(seg >= min(0.9, seg.max()))[0]
            leave = side_start + int(sat[-1]) + 1
            zc = next(i for i in range(leave, t + 1) if np.sign(a4[i]) == s)
            out.append({'leave': int(leave), 'zero_cross': int(zc), 'reach_other': int(t), 'dir': f'{side:+d}->{s:+d}'})
        if s != side:
            side_start = t
        side = s
    return out


def analyze(path):
    z = np.load(path)
    a4 = z['actions'][:, 3]
    valid = z['obj_valid'].astype(bool)
    sw = switches(a4)
    for s in sw:
        k = s['leave']
        s['valid_at_leave'] = bool(valid[k])
        prev_valid = np.nonzero(valid[:k + 1])[0]
        s['steps_since_valid_at_leave'] = int(k - prev_valid[-1]) if prev_valid.size else None
    inval = np.nonzero(~valid)[0]
    a0 = float(np.mean(a4[:3]))
    return {'file': path, 'steps': int(len(a4)), 'a4_start': a0, 'start_branch': '+' if a0 > 0 else '-',
            'switches': sw, 'n_switches': len(sw), 'first_invalid_step': int(inval[0]) if inval.size else None,
            'valid_rate': float(valid.mean()), 'tcp_z2_max': float(z['tcp_z2'].max()),
            'dmin_mm': float(1000 * z['dist'].min()), 'step_dmin': int(np.argmin(z['dist'])),
            'dfinal_mm': float(1000 * z['dist'][-1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='directorios con index.jsonl (se recorren recursivamente)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for run in args.runs:
        for idx in sorted(glob.glob(os.path.join(run, '**', 'index.jsonl'), recursive=True)):
            d = os.path.dirname(idx)
            for r in map(json.loads, open(idx)):
                a = analyze(os.path.join(d, r['file']))
                a.update({'run': d, 'seed': r['seed'], 'level': r['level_name'], 'success': r['success'],
                          'stalled': r['stalled']})
                rows.append(a)
    with open(os.path.join(args.out, 'branch_episodes.json'), 'w') as fh:
        json.dump(rows, fh, indent=1)
    lines = ['| grupo | episodios | éxitos | rama inicial − | con cambio de rama | éxitos con cambio | '
             'cambio iniciado con detección válida | TCP z máx > −0,5 |', '|---|---|---|---|---|---|---|---|']
    groups = {}
    for r in rows:
        groups.setdefault(r['run'], []).append(r)
    groups['TOTAL'] = rows
    for g, rs in groups.items():
        sw = [r for r in rs if r['n_switches']]
        firsts = [r['switches'][0] for r in sw]
        lines.append(f'| {g} | {len(rs)} | {sum(r["success"] for r in rs)} | '
                     f'{sum(r["start_branch"] == "-" for r in rs)} | {len(sw)} | {sum(r["success"] for r in sw)} | '
                     f'{sum(f["valid_at_leave"] for f in firsts)}/{len(firsts)} | '
                     f'{sum(r["tcp_z2_max"] > -0.5 for r in rs)} |')
    fails = [r for r in rows if not r['success']]
    lines += ['', '| episodio | nivel | rama inicial | cambios (deja saturación → cruza 0 → otra rama) | '
              'válida al iniciar | 1.ª pérdida | dmin (paso) | dfinal | TCP z máx |', '|---|---|---|---|---|---|---|---|---|']
    for r in fails:
        sws = '; '.join(f'{s["dir"]} {s["leave"]}→{s["zero_cross"]}→{s["reach_other"]}' for s in r['switches']) or '—'
        v = ', '.join(f'{s["valid_at_leave"]} ({s["steps_since_valid_at_leave"]})' for s in r['switches']) or '—'
        lines.append(f'| {os.path.basename(os.path.dirname(r["run"]))}/{r["seed"]} | {r["level"]} | {r["start_branch"]} | '
                     f'{sws} | {v} | {r["first_invalid_step"]} | {r["dmin_mm"]:.1f} ({r["step_dmin"]}) | '
                     f'{r["dfinal_mm"]:.1f} | {r["tcp_z2_max"]:+.2f} |')
    with open(os.path.join(args.out, 'branch_table.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
