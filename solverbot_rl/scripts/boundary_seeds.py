#!/usr/bin/env python3
"""Semillas de `reach_var_l` en las zonas límite de factibilidad de B0 (exp/solverbot-wrist-branch-consistency).

Clasifica la posición inicial del cubo de cada semilla de un banco con la regla de B0 sobre la grilla de factibilidad
(celda más cercana): `A_respaldo` (B no viable), `B30` (B solo a 30°) y `frontera_B20` (B@20 con alguna vecina de
otra opción). Toma las primeras N de cada zona. Es la misma regla usada para 921000+ y 931000+ en la Fase B, ahora
como script versionado. Solo lee la configuración del nivel; no ejecuta políticas.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diagnose_wrist_branches as W  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feasibility', default='runs/wrist_branches/feasibility/feasibility.json')
    ap.add_argument('--pool', required=True, help='p. ej. 961000-961999')
    ap.add_argument('--per-zone', type=int, default=30)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    F = json.load(open(a.feasibility))
    cells = {}
    for r in F['rows']:
        if r['map'] != 'cube_home':
            continue
        cells.setdefault((round(r['dx'], 3), round(r['dy'], 3)), {})[(r['family'], r['tilt'])] = r['viable']
    choice = {k: next((c for c in (('B', 20), ('B', 30), ('A', 20), ('A', 30)) if v[c]), None)
              for k, v in cells.items()}
    keys = np.array(list(choice))
    step = 0.014

    def zone(dx, dy):
        i = np.argmin(((keys - [dx, dy]) ** 2).sum(1))
        k = tuple(keys[i])
        c = choice[k]
        neigh = [choice.get((round(k[0] + x, 3), round(k[1] + y, 3))) for x in (-step, 0, step) for y in (-step, 0, step)]
        neigh = [n for n in neigh if n is not None]
        if c == ('A', 20):
            return 'A_respaldo'
        if c == ('B', 30):
            return 'B30'
        if any(n != c for n in neigh):
            return 'frontera_B20'
        return None

    env = W.make_env()
    u = env.unwrapped
    li = W.level_index(env, 'reach_var_l')
    nominal = np.array(u.cfg['world']['object']['xy'])
    sel = collections.defaultdict(list)
    for seed in W.parse_seeds(a.pool):
        env.reset(seed=seed, options={'level': li})
        z = zone(*(np.array(u.start_obj_xy) - nominal))
        if z and len(sel[z]) < a.per_zone:
            sel[z].append(seed)
    print({k: len(v) for k, v in sel.items()})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(sel, open(a.out, 'w'))
    for k, v in sel.items():
        print(f'{k}=' + ','.join(map(str, v)))


if __name__ == '__main__':
    main()
