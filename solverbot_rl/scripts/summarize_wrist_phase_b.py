#!/usr/bin/env python3
"""Resumen versionable de la Fase B (exp/solverbot-wrist-branch-consistency).

Lee los rollouts cinemáticos (index.jsonl + episodes/*.npz) de una evaluación y, si existe, el análisis de ramas
(analysis/episodes.jsonl), y escribe un JSON compacto con, por política y escenario:
éxito (IC95 Wilson), distancia final del TCP, pasos, tasa de cambio de acción (continuidad articular), margen mínimo
a los límites articulares, tasa de detección del cubo, colisiones, estancamientos, atascos en el límite de muñeca y
cambios de rama; y pruebas de McNemar exactas pareadas por semilla contra una política de referencia.

Solo lee resultados; no ejecuta políticas ni toca el entorno.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def scenario_of(run, policy):
    """Nombre del escenario a partir del directorio <backend>_<política>_<escenario> (cinemático o Gazebo)."""
    return run.split(f'_{policy}_', 1)[1]


def load(root: Path):
    rows = []
    for d in sorted((root / 'rollouts').iterdir()):
        idx = d / 'index.jsonl'
        if not idx.exists():
            continue
        for line in idx.open():
            r = json.loads(line)
            z = np.load(d / r['file'])
            lo, hi = np.array(r['lo']), np.array(r['hi'])
            q = lo + (z['obs'][:, :6] + 1.0) * (hi - lo) / 2.0   # obs normalizada a [-1, 1]
            margin = np.minimum(q - lo, hi - q).min()
            a = z['actions'][:, :6]
            r['_scenario'] = scenario_of(d.name, r['policy'])
            r['_margin_min'] = float(margin)
            r['_dq_max'] = float(np.abs(a).max() * r['delta'])
            rows.append(r)
    ana = {}
    af = root / 'analysis' / 'episodes.jsonl'
    if af.exists():
        for line in af.open():
            e = json.loads(line)
            ana[(e['policy'], e['run'] if 'run' in e else None, e['seed'])] = e
    return rows, ana


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--ref', default='bc_v2w')
    a = ap.parse_args()
    rows, ana = load(a.root)
    by_run = {}
    if ana:
        for (pol, run, seed), e in ana.items():
            by_run[(pol, scenario_of(run, pol) if run else None, seed)] = e

    groups = defaultdict(list)
    for r in rows:
        groups[(r['policy'], r['_scenario'])].append(r)
    table = {}
    for (pol, sc), rs in sorted(groups.items()):
        n = len(rs)
        k = sum(r['success'] for r in rs)
        es = [by_run.get((pol, sc, r['seed'])) for r in rs]
        es = [e for e in es if e is not None]
        ent = {
            'n': n, 'success': k, 'success_ci95': wilson(k, n),
            'dfinal_mm_mean': round(float(np.mean([r['dfinal_mm'] for r in rs])), 1),
            'length_mean': round(float(np.mean([r['length'] for r in rs])), 1),
            'detection_rate_mean': round(float(np.mean([r['valid_rate'] for r in rs])), 3),
            'collisions': int(sum(r['collision'] for r in rs)),
            'stalled': int(sum(r['stalled'] for r in rs)),
            'joint_margin_min_rad_p05': round(float(np.percentile([r['_margin_min'] for r in rs], 5)), 3),
        }
        if es:
            ent.update({
                'action_rate_mean': round(float(np.mean([e['action_rate'] for e in es])), 3),
                'executed_final': {f: sum(e['executed_final'] == f for e in es) for f in ('A', 'B', None)},
                'episodes_real_branch_change': sum(e['n_real_changes'] > 0 for e in es),
                'episodes_intention_switch': sum(e['n_intention_switches'] > 0 for e in es),
                'failures_stuck_at_wrist_limit': sum((not e['success']) and e['stuck_at_wrist_limit'] for e in es),
                'failures_intention_switch': sum((not e['success']) and e['n_intention_switches'] > 0 for e in es),
            })
            ent['executed_final'] = {str(k_): v for k_, v in ent['executed_final'].items()}
        table.setdefault(pol, {})[sc] = ent

    # McNemar pareado por semilla contra la referencia
    succ = {(r['policy'], r['_scenario'], r['seed']): bool(r['success']) for r in rows}
    tests = {}
    for pol in table:
        if pol == a.ref:
            continue
        tests[pol] = {}
        scs = sorted(table[pol])
        groups_sc = {'total_sin_reach': [s for s in scs if s != 'reach'], **{s: [s] for s in scs}}
        for name, sl in groups_sc.items():
            b = c = 0
            for (p_, s_, seed), v in succ.items():
                if p_ != pol or s_ not in sl:
                    continue
                w = succ.get((a.ref, s_, seed))
                if w is None:
                    continue
                b += v and not w      # gana la candidata
                c += w and not v      # gana la referencia
            tests[pol][name] = {'cand_only': b, 'ref_only': c, 'p': float(f'{mcnemar_exact(b, c):.3g}')}
    totals = {pol: {'success': sum(v['success'] for s, v in t.items() if s != 'reach'),
                    'n': sum(v['n'] for s, v in t.items() if s != 'reach')} for pol, t in table.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({'root': str(a.root), 'reference': a.ref, 'table': table, 'totals_sin_reach': totals,
               'mcnemar_vs_ref': tests}, a.out.open('w'), indent=1)
    for pol, t in table.items():
        print(pol, totals[pol], {s: f"{v['success']}/{v['n']}" for s, v in t.items()})
    for pol, t in tests.items():
        print('mcnemar', pol, {s: (v['cand_only'], v['ref_only'], v['p']) for s, v in t.items()})


if __name__ == '__main__':
    main()
