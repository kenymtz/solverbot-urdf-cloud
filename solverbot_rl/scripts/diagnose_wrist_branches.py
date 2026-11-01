#!/usr/bin/env python3
"""Fase A — diagnóstico de las ramas de muñeca en `reach` (exp/solverbot-wrist-branch-consistency).

Solo diagnóstico: no entrena, no modifica el experto, el entorno ni las políticas. La verdad del simulador se usa
únicamente para MEDIR (soluciones de IK de referencia de cada episodio); las políticas ven solo la observación.

Subcomandos:
  ikmap      mapas de la rama que elige ScriptedExpert (su propia _ik) según la posición del cubo y el estado
             inicial, con el coste de cada familia de soluciones
  rollouts   episodios de scripted | bc_v1 | bc_v2w (kinematic o gazebo) con registro por paso; para el experto,
             además su objetivo articular q_goal
  analyze    rama elegida / intentada / ejecutada, cambios de signo frente a cambios reales, contradicciones entre
             etiquetas y promediado de BC
  plots      figuras

    python3 solverbot_rl/scripts/diagnose_wrist_branches.py ikmap --out runs/wrist_branches/ikmap
    python3 solverbot_rl/scripts/diagnose_wrist_branches.py rollouts --policy scripted --level reach_var_m \
        --seeds 915400-915599 --out runs/wrist_branches/rollouts
    python3 solverbot_rl/scripts/diagnose_wrist_branches.py analyze --root runs/wrist_branches \
        --out solverbot_rl/docs/resultados/wrist_branches
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solverbot_rl.envs.pick_place import SolverBotPickPlaceEnv  # noqa: E402
from solverbot_rl.imitation import branch_analysis as B  # noqa: E402
from solverbot_rl.imitation import dataset as D  # noqa: E402
from solverbot_rl.imitation.experiment import env_kwargs, level_index, load_experiment, parse_seeds  # noqa: E402
from solverbot_rl.imitation.rollout import run_episode  # noqa: E402

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'resultados',
                          'reach_bc_v1', 'modelos')
MODELS = {'bc_v1': os.path.join(_MODEL_DIR, 'bc_v1.zip'), 'bc_v2w': os.path.join(_MODEL_DIR, 'bc_v2w.zip')}


def make_env(backend='kinematic', experiment='reach_bc_v1'):
    exp = load_experiment(experiment)
    return SolverBotPickPlaceEnv(backend=backend, **env_kwargs(exp))


def pregrasp_point(env, obj_base):
    return np.asarray(obj_base, float) + np.array([0.0, 0.0, env.unwrapped.params.pregrasp_height])


def expert_choice(env, expert, q0, p):
    """Objetivo articular que elegiría ScriptedExpert (su propio _ik, sin modificar) desde q0 hacia p."""
    u = env.unwrapped
    u._q_cmd = np.asarray(q0, float).copy()
    expert.reset()
    return np.asarray(expert._ik(np.asarray(p, float)), float)


# ============================================================================================================ ikmap
def cmd_ikmap(args):
    from solverbot_rl.expert import ScriptedExpert
    env = make_env()
    u = env.unwrapped
    ex = ScriptedExpert(env)
    env.reset(seed=915000, options={'level': level_index(env, 'reach')})       # cubo nominal, home exacto
    home = np.asarray(u.cfg['robot']['home'], float)
    obj0 = np.asarray(u.backend.object_pose_truth(), float)
    nominal_world = np.asarray(u.cfg['world']['object']['xy'], float)
    base0 = np.asarray(u.backend.world_xy_to_base(nominal_world), float)
    rows = []
    # 1) posición del cubo (±7 cm, como reach_var_l) con home exacto
    g = np.linspace(-0.07, 0.07, args.grid)
    for dx in g:
        for dy in g:
            xy = np.asarray(u.backend.world_xy_to_base(nominal_world + [dx, dy]), float)
            obj = np.array([xy[0], xy[1], obj0[2]])
            rows.append(_ik_row(env, ex, home, obj, {'map': 'cube', 'dx': float(dx), 'dy': float(dy)}))
        print(f'[ikmap] cubo dx={dx:+.3f}', flush=True)
    # 2) estado inicial: joint4 y joint6 de q0 (los dos ejes del empate) con el cubo nominal
    for d4 in np.linspace(-0.15, 0.15, args.grid):
        for d6 in np.linspace(-0.15, 0.15, args.grid):
            q0 = home.copy()
            q0[3] += d4
            q0[5] += d6
            rows.append(_ik_row(env, ex, q0, obj0, {'map': 'q0_j4_j6', 'd4': float(d4), 'd6': float(d6)}))
        print(f'[ikmap] q0 d4={d4:+.3f}', flush=True)
    # 3) escenarios reales de cada nivel (q0 con ruido + cubo), mismas semillas que los rollouts del diagnóstico
    for lvl, spec in (('reach_var_s', args.seeds_s), ('reach_var_m', args.seeds_m), ('reach_var_l', args.seeds_l)):
        li = level_index(env, lvl)
        for seed in parse_seeds(spec):
            env.reset(seed=seed, options={'level': li})
            rows.append(_ik_row(env, ex, u.start_q, u.backend.object_pose_truth(),
                                {'map': 'level', 'level': lvl, 'seed': seed}))
        print(f'[ikmap] {lvl} listo', flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'ikmap.json'), 'w') as fh:
        json.dump({'home': home.tolist(), 'base_nominal': base0.tolist(), 'rows': rows}, fh)
    print('[ikmap] filas', len(rows))


def _ik_row(env, ex, q0, obj, meta):
    u = env.unwrapped
    p = pregrasp_point(env, obj)
    qe = expert_choice(env, ex, q0, p)
    sols = B.ik_solution_set(env, q0, p, n_starts=12)
    fc = B.family_costs(sols, q0, u.delta)
    row = dict(meta)
    row.update({'q0': [float(v) for v in q0], 'expert_q': qe.tolist(), 'expert_family': B.family_of(qe),
                'tilt': fc['tilt'], 'n_solutions': len(sols)})
    for f in B.FAMILIES:
        c = fc[f]
        row[f] = None if c is None else {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()}
    if fc['A'] and fc['B']:
        row['min_l2_family'] = 'A' if fc['A']['l2'] < fc['B']['l2'] else 'B'
        row['min_steps_family'] = 'A' if fc['A']['steps'] < fc['B']['steps'] else 'B'
        row['dl2_A_minus_B'] = fc['A']['l2'] - fc['B']['l2']
        row['dsteps_A_minus_B'] = fc['A']['steps'] - fc['B']['steps']
    return row


# ========================================================================================================= rollouts
class _LoggedExpert:
    """Envoltura de SOLO LECTURA: llama a ScriptedExpert.act() sin cambiarlo y registra su q_goal y su fase."""

    def __init__(self, expert):
        self.ex, self.q_goal, self.phase = expert, [], []

    def reset(self):
        self.ex.reset()
        self.q_goal, self.phase = [], []

    def act(self):
        a = self.ex.act()
        qg = self.ex.q_goal
        self.q_goal.append(np.full(6, np.nan) if qg is None else np.asarray(qg, float))
        self.phase.append(self.ex.phase)
        return a


def cmd_rollouts(args):
    env = make_env(args.backend)
    li = level_index(env, args.level)
    expert, act = None, None
    if args.policy == 'scripted':
        from solverbot_rl.expert import ScriptedExpert
        expert = _LoggedExpert(ScriptedExpert(env))
    else:
        from solverbot_rl.imitation.bc import load_policy
        act = load_policy(MODELS[args.policy], deterministic=True)
    out = os.path.join(args.out, f'{args.backend}_{args.policy}_{args.level}' + (f'_{args.tag}' if args.tag else ''))
    os.makedirs(out, exist_ok=True)
    done = {r['seed'] for r in D.read_index(out)}
    seeds = [s for s in parse_seeds(args.seeds) if s not in done]
    print(f'[rollouts] {args.policy} {args.level} {args.backend} {len(seeds)} episodios -> {out}', flush=True)
    for i, seed in enumerate(seeds):
        arrays, s = run_episode(env, seed, li, act=act, expert=expert, rng=np.random.default_rng(seed))
        if expert is not None:
            arrays['expert_q_goal'] = np.asarray(expert.q_goal, float)
        s.update({'policy': args.policy, 'backend': args.backend, 'expert_in_loop': expert is not None,
                  'lo': env.unwrapped.lo.tolist(), 'hi': env.unwrapped.hi.tolist(),
                  'delta': float(env.unwrapped.delta), 'pregrasp_height': float(env.unwrapped.params.pregrasp_height)})
        D.save_episode(out, arrays, s)
        print(f'[rollouts] {i + 1}/{len(seeds)} seed={seed} éxito={s["success"]} pasos={s["length"]} '
              f'dfin={s["dfinal_mm"]:.1f} zf={s["tcp_z2_final"]:.2f} det={s["valid_rate"]:.2f}', flush=True)
    env.close()


# ========================================================================================================== analyze
def _load_runs(root):
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, 'rollouts', '*'))):
        rows = D.read_index(d)
        if rows:
            runs[os.path.basename(d)] = (d, sorted(rows, key=lambda r: r['seed']))
    return runs


def _episode_arrays(d, seed):
    return dict(np.load(D.episode_file(d, seed), allow_pickle=False))


def _reps_cache(env, cache, s):
    key = (s['level'], s['seed'])
    if key not in cache:
        q0 = np.asarray(s['start_q'], float)
        p = np.asarray(s['obj_start_truth'], float) + np.array([0, 0, s['pregrasp_height']])
        fc = B.family_costs(B.ik_solution_set(env, q0, p, n_starts=12), q0, s['delta'])
        cache[key] = fc
    return cache[key]


def episode_branch_record(arr, s, fc):
    lo, hi = np.asarray(s['lo']), np.asarray(s['hi'])
    q = B.denorm_q(arr['obs'], lo, hi)
    a = arr['actions']
    feasible = {f: fc[f] is not None for f in B.FAMILIES}
    ra = np.asarray(fc['A']['q']) if feasible['A'] else B.mirrored_rep(fc['B']['q'])
    rb = np.asarray(fc['B']['q']) if feasible['B'] else B.mirrored_rep(fc['A']['q'])
    aff = np.array([B.affinity(qq, ra, rb) for qq in q])
    lab, changes = B.executed_branch(aff, dists=B.wrist_distances(q, ra, rb))
    intent = [B.intended_branch(qq, aa, ra, rb) for qq, aa in zip(q, a)]
    sc = B.sign_changes(a[:, 3])
    far = np.asarray(arr['dist']) > 0.045                     # fase lejana: la orden de muñeca decide la rama
    isw = B.intention_switches(intent, active=far)
    valid = arr['obs'][:, 16] > 0.5
    rec = {'seed': s['seed'], 'level': s['level_name'], 'policy': s['policy'], 'backend': s['backend'],
           'success': s['success'], 'length': s['length'], 'dmin_mm': s['dmin_mm'], 'dfinal_mm': s['dfinal_mm'],
           'tcp_z2_final': s['tcp_z2_final'], 'tcp_z2_max': float(np.max(arr['tcp_z2'])),
           'stalled': s['stalled'], 'collision': s['collision'], 'valid_rate': s['valid_rate'],
           'wall_s': s['wall_s'], 'tilt': fc['tilt'], 'feasible_A': feasible['A'], 'feasible_B': feasible['B'],
           'other_tilt_A': bool(feasible['A'] and fc['A']['other_tilt']),
           'other_tilt_B': bool(feasible['B'] and fc['B']['other_tilt']),
           'cost_A_l2': fc['A']['l2'] if feasible['A'] else None, 'cost_B_l2': fc['B']['l2'] if feasible['B'] else None,
           'cost_A_steps': fc['A']['steps'] if feasible['A'] else None,
           'cost_B_steps': fc['B']['steps'] if feasible['B'] else None,
           'q0': [float(v) for v in s['start_q']], 'obj_xy_cmd': s['obj_xy_cmd'],
           'a0': [float(v) for v in a[0, :6]], 'intent_t0': intent[0],
           'limit_joints_final': [int(j) for j in np.nonzero(np.any(
               (q[-10:] <= lo + 0.02) | (q[-10:] >= hi - 0.02), axis=0))[0]],
           'executed_final': lab[-1], 'real_branch_changes': changes, 'n_real_changes': len(changes),
           'a4_sign_changes': sc, 'n_a4_sign_changes': len(sc),
           'intention_switches': isw, 'n_intention_switches': len(isw),
           'intent_first': next((f for f in intent if f is not None), None),
           'ambiguous_intent_frac': float(np.mean([f is None for f in intent])),
           'valid_at_real_changes': [bool(valid[t]) for t in changes],
           'valid_at_intention_switches': [bool(valid[t]) for t in isw],
           'valid_in_5_before_switch': [bool(valid[max(0, t - 5):t + 1].all()) for t in isw],
           'action_rate': float(np.mean(np.sum(np.diff(a[:, :6], axis=0) ** 2, axis=1))) if len(a) > 1 else 0.0,
           'detour_mm': B.detour_mm(arr['dist']), 'tcp_z2_max_far': float(np.max(arr['tcp_z2'][far])) if far.any()
           else None, 'wrist_limit_frac': B.wrist_limit_frac(q, lo, hi),
           'stuck_at_wrist_limit': bool(B.wrist_limit_frac(q[-10:], lo, hi) > 0.8)}
    if 'expert_q_goal' in arr and len(arr['expert_q_goal']):
        qg = arr['expert_q_goal']
        fams = [B.family_of(g) for g in qg if np.all(np.isfinite(g))]
        rec['expert_family'] = fams[0] if fams else None
        rec['expert_goal_family_changes'] = int(sum(1 for x, y in zip(fams[:-1], fams[1:]) if x != y))
    return rec, {'q': q, 'a': a, 'aff': aff, 'valid': valid, 'tcp_z2': arr['tcp_z2'], 'dist': arr['dist']}


def cmd_analyze(args):
    env = make_env()
    runs = _load_runs(args.root)
    cache = {}
    recs, traces = [], {}
    for name, (d, rows) in runs.items():
        for s in rows:
            arr = _episode_arrays(d, s['seed'])
            fc = _reps_cache(env, cache, s)
            if fc['A'] is None and fc['B'] is None:
                continue
            rec, tr = episode_branch_record(arr, s, fc)
            rec['run'] = name
            recs.append(rec)
            traces[(name, s['seed'])] = tr
        print(f'[analyze] {name}: {len(rows)} episodios', flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'episodes.jsonl'), 'w') as fh:
        for r in recs:
            fh.write(json.dumps(r) + '\n')
    summary = summarize(recs)
    summary['contradictions'] = contradictions(runs, cache, env, args)
    with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1, default=lambda o: None)
    print(json.dumps({k: v for k, v in summary.items() if k != 'contradictions'}, indent=1, default=str)[:4000])


def _wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(c - h, 3), round(c + h, 3))


def summarize(recs):
    out = {}
    groups = {}
    for r in recs:
        groups.setdefault(r['run'], []).append(r)
    for name, rs in groups.items():
        n = len(rs)
        k = sum(r['success'] for r in rs)
        g = {'n': n, 'success': k, 'success_ci95': _wilson(k, n),
             'executed_final': {f: sum(r['executed_final'] == f for r in rs) for f in ('A', 'B', None)},
             'episodes_with_real_change': sum(r['n_real_changes'] > 0 for r in rs),
             'episodes_with_a4_sign_change': sum(r['n_a4_sign_changes'] > 0 for r in rs),
             'episodes_with_intention_switch': sum(r['n_intention_switches'] > 0 for r in rs),
             'sign_change_without_real_change': sum(r['n_a4_sign_changes'] > 0 and r['n_real_changes'] == 0
                                                    for r in rs),
             'failures_with_real_change': sum((not r['success']) and r['n_real_changes'] > 0 for r in rs),
             'failures_with_intention_switch': sum((not r['success']) and r['n_intention_switches'] > 0 for r in rs),
             'failures': n - k,
             'intention_switch_valid_frac': _frac([v for r in rs for v in r['valid_at_intention_switches']]),
             'intention_switch_valid5_frac': _frac([v for r in rs for v in r['valid_in_5_before_switch']]),
             'real_change_valid_frac': _frac([v for r in rs for v in r['valid_at_real_changes']]),
             'ambiguous_intent_frac_mean': float(np.mean([r['ambiguous_intent_frac'] for r in rs])),
             'tcp_z2_final_mean': float(np.mean([r['tcp_z2_final'] for r in rs])),
             'detour_mm_mean': float(np.mean([r['detour_mm'] for r in rs])),
             'episodes_detour_gt_30mm': sum(r['detour_mm'] > 30 for r in rs),
             'episodes_tool_tilt_lost': sum((r['tcp_z2_max_far'] or -1) > -0.6 for r in rs),
             'episodes_stuck_at_wrist_limit': sum(r['stuck_at_wrist_limit'] for r in rs),
             'failures_stuck_at_wrist_limit': sum((not r['success']) and r['stuck_at_wrist_limit'] for r in rs),
             'failures_with_detour': sum((not r['success']) and r['detour_mm'] > 30 for r in rs),
             'dfinal_mm_mean': float(np.mean([r['dfinal_mm'] for r in rs])),
             'length_mean': float(np.mean([r['length'] for r in rs])),
             'wall_s_mean': float(np.mean([r['wall_s'] for r in rs]))}
        g['success_by_executed_family'] = {f: [sum(r['success'] for r in rs if r['executed_final'] == f),
                                               sum(r['executed_final'] == f for r in rs)] for f in ('A', 'B')}
        g['success_both_feasible'] = [sum(r['success'] for r in rs if r['feasible_A'] and r['feasible_B']),
                                      sum(r['feasible_A'] and r['feasible_B'] for r in rs)]
        g['success_single_family'] = [sum(r['success'] for r in rs if not (r['feasible_A'] and r['feasible_B'])),
                                      sum(not (r['feasible_A'] and r['feasible_B']) for r in rs)]
        lim = {}
        for r in rs:
            if not r['success']:
                for j in r['limit_joints_final']:
                    lim[f'joint{j + 1}'] = lim.get(f'joint{j + 1}', 0) + 1
        g['failures_joint_at_limit'] = lim
        if 'expert_family' in rs[0]:
            g['expert_family'] = {f: sum(r.get('expert_family') == f for r in rs) for f in ('A', 'B')}
            g['expert_goal_family_changes'] = sum(r.get('expert_goal_family_changes', 0) > 0 for r in rs)
            ch = [r for r in rs if r.get('expert_family') and r['feasible_A'] and r['feasible_B']]
            g['single_family_scenarios'] = sum(not (r['feasible_A'] and r['feasible_B']) for r in rs)
            g['expert_matches_min_l2'] = sum((r['expert_family'] == ('A' if r['cost_A_l2'] < r['cost_B_l2'] else 'B'))
                                             for r in ch)
            g['expert_matches_min_steps'] = sum((r['expert_family'] == ('A' if r['cost_A_steps'] < r['cost_B_steps']
                                                                         else 'B')) for r in ch)
        out[name] = g
    return out


def _frac(v):
    return None if not v else round(float(np.mean(v)), 3)


def contradictions(runs, cache, env, args):
    """Etiquetas del experto: pares de observaciones cercanas (entrada del actor, estandarizada) de episodios
    distintos con familias opuestas; y predicciones de bc_v1 / bc_v2w sobre esos mismos estados."""
    from solverbot_rl.imitation.bc import load_policy
    X, fam, ep, a4, obs_all, act_all, t_all = [], [], [], [], [], [], []
    for name, (d, rows) in runs.items():
        if not name.startswith('kinematic_scripted'):
            continue
        for s in rows:
            arr = _episode_arrays(d, s['seed'])
            if 'expert_q_goal' not in arr:
                continue
            fams = [B.family_of(g) for g in arr['expert_q_goal'] if np.all(np.isfinite(g))]
            if not fams:
                continue
            f = fams[0]
            n = len(arr['obs'])
            X.append(B.actor_features(arr['obs']))
            fam += [f] * n
            ep += [f'{name}:{s["seed"]}'] * n
            a4.append(arr['actions'][:, 3])
            obs_all.append(arr['obs'])
            act_all.append(arr['actions'])
            t_all.append(np.arange(n))
    if not X:
        return {}
    X = np.concatenate(X)
    fam = np.array(fam)
    ep = np.array(ep)
    a4 = np.concatenate(a4)
    obs_all = np.concatenate(obs_all)
    act_all = np.concatenate(act_all)
    t_all = np.concatenate(t_all)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    out = {'n_steps': int(len(X)), 'n_episodes': int(len(set(ep.tolist()))),
           'family_steps': {f: int((fam == f).sum()) for f in ('A', 'B')}}
    # distancia al vecino más cercano de OTRO episodio: escala de referencia del radio
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xs), size=min(2000, len(Xs)), replace=False)
    d2 = (Xs[idx] ** 2).sum(1)[:, None] + (Xs ** 2).sum(1)[None, :] - 2 * Xs[idx] @ Xs.T
    d2[ep[idx][:, None] == ep[None, :]] = np.inf
    nn = np.sqrt(np.maximum(d2.min(1), 0))
    out['nn_other_episode_quantiles'] = {q: float(np.quantile(nn, q / 100)) for q in (10, 25, 50, 75)}
    out['by_radius'] = {}
    conflict_mask = None
    for qk in (10, 25, 50):
        r = out['nn_other_episode_quantiles'][qk]
        res = B.label_conflicts(Xs, fam, ep, a4, r)
        m = res.pop('has_conflict')
        if qk == 25:
            conflict_mask = m
        out['by_radius'][f'p{qk}'] = {'radius': r, **res}
    # primer paso: estados iniciales casi iguales con familias opuestas
    first = t_all == 0
    res0 = B.label_conflicts(Xs[first], fam[first], ep[first], a4[first], out['nn_other_episode_quantiles'][50])
    conflict0 = np.zeros(len(Xs), bool)
    conflict0[np.nonzero(first)[0]] = res0.pop('has_conflict')
    out['first_step'] = res0
    # predicciones de BC sobre los estados del experto
    preds = {}
    for name, path in MODELS.items():
        act = load_policy(path, deterministic=True)
        P = np.stack([act(o) for o in obs_all])
        preds[name] = P
        e4, p4 = act_all[:, 3], P[:, 3]
        sat = np.abs(e4) > 0.9
        r_conf = conflict_mask & sat
        r_pure = (~conflict_mask) & sat
        out[f'bc_{name}'] = {
            'mse_wrist': float(np.mean((P[:, 3:5] - act_all[:, 3:5]) ** 2)),
            'intermediate_frac_all_saturated': float(np.mean(np.abs(p4[sat]) < 0.5)),
            'intermediate_frac_conflict': float(np.mean(np.abs(p4[r_conf]) < 0.5)) if r_conf.any() else None,
            'intermediate_frac_pure': float(np.mean(np.abs(p4[r_pure]) < 0.5)) if r_pure.any() else None,
            'opposite_sign_frac_conflict': float(np.mean(np.sign(p4[r_conf]) != np.sign(e4[r_conf])))
            if r_conf.any() else None,
            'opposite_sign_frac_pure': float(np.mean(np.sign(p4[r_pure]) != np.sign(e4[r_pure]))) if r_pure.any() else None,
            'n_conflict_saturated': int(r_conf.sum()), 'n_pure_saturated': int(r_pure.sum()),
            'abs_pred_a4_conflict_mean': float(np.mean(np.abs(p4[r_conf]))) if r_conf.any() else None,
            'abs_pred_a4_pure_mean': float(np.mean(np.abs(p4[r_pure]))) if r_pure.any() else None,
        }
        # primer paso (donde se decide la rama): sobre TODOS los estados iniciales y sobre los que tienen un vecino
        # inicial con la familia opuesta
        f0 = first
        c0 = conflict0 & first
        lab_sign = np.where(fam == 'A', 1.0, -1.0)
        out[f'bc_{name}']['t0'] = {
            'n': int(f0.sum()), 'n_conflict': int(c0.sum()),
            'agree_with_expert_family': float(np.mean(np.sign(p4[f0]) == lab_sign[f0])),
            'agree_with_expert_family_conflict': float(np.mean(np.sign(p4[c0]) == lab_sign[c0])) if c0.any() else None,
            'intermediate_frac': float(np.mean(np.abs(p4[f0]) < 0.5)),
            'intermediate_frac_conflict': float(np.mean(np.abs(p4[c0]) < 0.5)) if c0.any() else None,
            'pred_sign_A_frac': float(np.mean(p4[f0] > 0)), 'expert_A_frac': float(np.mean(fam[f0] == 'A'))}
    np.savez_compressed(os.path.join(args.out, 'contradiction_arrays.npz'), a4_expert=act_all[:, 3],
                        a5_expert=act_all[:, 4], fam=fam, t=t_all, conflict=conflict_mask, conflict0=conflict0,
                        **{f'a4_{k}': v[:, 3] for k, v in preds.items()}, **{f'a5_{k}': v[:, 4] for k, v in preds.items()})
    return out


# ======================================================================================================= labelrules
RULES = {
    'experto (IK global actual)': None,
    'mínimo L2 (todas las soluciones)': lambda r: 'A' if r['cost_A_l2'] < r['cost_B_l2'] else 'B',
    'mínimo tiempo (pasos)': lambda r: 'A' if r['cost_A_steps'] < r['cost_B_steps'] else 'B',
    'preferir A salvo infactible': lambda r: 'A',
    'preferir B salvo infactible': lambda r: 'B',
}


def cmd_labelrules(args):
    """Coherencia de ETIQUETAS (no se ejecuta ningún etiquetador ni se entrena): para cada regla de elección de
    familia, tasa de pares de estados INICIALES cercanos (entrada del actor, estandarizada) con familias distintas.
    La regla solo puede elegir entre familias factibles."""
    eps = {(r['level'], r['seed']): r for r in map(json.loads, open(os.path.join(args.analysis, 'episodes.jsonl')))
           if r['policy'] == 'scripted' and r['backend'] == 'kinematic'}
    runs = _load_runs(args.root)
    X, keys = [], []
    for name, (d, rows) in runs.items():
        if not name.startswith('kinematic_scripted'):
            continue
        for s in rows:
            k = (s['level_name'], s['seed'])
            if k not in eps:
                continue
            arr = _episode_arrays(d, s['seed'])
            X.append(B.actor_features(arr['obs'][0]))
            keys.append(k)
    X = np.array(X)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    ep = np.array([f'{a}:{b}' for a, b in keys])
    d2 = (Xs ** 2).sum(1)[:, None] + (Xs ** 2).sum(1)[None, :] - 2 * Xs @ Xs.T
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(np.maximum(d2.min(1), 0))
    out = {'n_initial_states': len(X), 'radius_quantiles_nn': {q: float(np.quantile(nn, q / 100)) for q in (50, 75, 90)}}
    for name, rule in RULES.items():
        fam = []
        for k in keys:
            r = eps[k]
            if not (r['feasible_A'] and r['feasible_B']):
                fam.append('A' if r['feasible_A'] else 'B')
            elif rule is None:
                fam.append(r['expert_family'])
            else:
                fam.append(rule(r))
        fam = np.array(fam)
        res = {}
        for q in (50, 75, 90):
            rad = out['radius_quantiles_nn'][q]
            c = B.label_conflicts(Xs, fam, ep, np.where(fam == 'A', 1.0, -1.0), rad)
            c.pop('has_conflict')
            res[f'p{q}'] = {'conflict_rate': round(c['conflict_rate'], 4), 'pairs': c['pairs_conflict'] + c['pairs_same']}
        # tiempo extra respecto del óptimo en tiempo y en L2 (costo de seguir la regla)
        extra_steps = [eps[k]['cost_' + f + '_steps'] - min(eps[k]['cost_A_steps'], eps[k]['cost_B_steps'])
                       for k, f in zip(keys, fam) if eps[k]['feasible_A'] and eps[k]['feasible_B']]
        out[name] = {'A_frac': float(np.mean(fam == 'A')), 'conflicts': res,
                     'extra_steps_mean': float(np.mean(extra_steps)), 'extra_steps_max': float(np.max(extra_steps))}
        print(name, out[name], flush=True)
    with open(os.path.join(args.analysis, 'label_rules.json'), 'w') as fh:
        json.dump(out, fh, indent=1)


# ============================================================================================================ plots
def cmd_plots(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(args.out, exist_ok=True)
    ik = json.load(open(os.path.join(args.root, 'ikmap', 'ikmap.json')))
    rows = ik['rows']
    col = {'A': '#d1495b', 'B': '#00798c'}
    # mapa por posición del cubo (home exacto)
    cube = [r for r in rows if r['map'] == 'cube']
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    sc = None
    for r in cube:
        single = not (r['A'] and r['B'])
        ax[0].scatter(100 * r['dx'], 100 * r['dy'], c=col[r['expert_family']], s=55, marker='s',
                      edgecolors='k' if single else 'none', linewidths=1.2)
        if r.get('dl2_A_minus_B') is not None:
            sc = ax[1].scatter(100 * r['dx'], 100 * r['dy'], c=r['dl2_A_minus_B'], cmap='coolwarm', vmin=-1, vmax=1,
                               s=55, marker='s')
            ax[2].scatter(100 * r['dx'], 100 * r['dy'], c=col[r['min_l2_family']], s=55, marker='s')
        else:
            for a in ax[1:]:
                a.scatter(100 * r['dx'], 100 * r['dy'], c='#bbbbbb', s=55, marker='x')
    if sc is not None:
        fig.colorbar(sc, ax=ax[1], shrink=0.8)
    ax[0].set_title('Rama ELEGIDA por el experto (home exacto)\nrojo A: joint4>0 · azul B: joint4<0 · borde: solo 1 familia a 20°')
    ax[1].set_title('Coste L2(A) - L2(B) desde home [rad]\n(<0: A más barata)')
    ax[2].set_title('Rama de MENOR coste L2 (x gris: solo existe una)')
    for a in ax:
        a.set_xlabel('dx cubo [cm]')
        a.set_ylabel('dy cubo [cm]')
        a.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'map_cube_home.png'), dpi=110)
    plt.close(fig)
    # mapa por estado inicial (joint4, joint6)
    q0m = [r for r in rows if r['map'] == 'q0_j4_j6']
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
    for r in q0m:
        ax[0].scatter(r['d4'], r['d6'], c=col[r['expert_family']], s=55, marker='s')
        if r.get('dl2_A_minus_B') is not None:
            ax[1].scatter(r['d4'], r['d6'], c=r['dl2_A_minus_B'], cmap='coolwarm', vmin=-0.5, vmax=0.5, s=55,
                          marker='s')
    ax[0].set_title('Rama elegida por el experto\nvs perturbación de joint4 y joint6 iniciales (cubo nominal)')
    ax[1].set_title('L2(A) - L2(B)')
    for a in ax:
        a.set_xlabel('Δ joint4 inicial [rad]')
        a.set_ylabel('Δ joint6 inicial [rad]')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'map_q0_j4_j6.png'), dpi=110)
    plt.close(fig)
    # escenarios reales: rama elegida vs coste
    lv = [r for r in rows if r['map'] == 'level' and r.get('dl2_A_minus_B') is not None]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for r in lv:
        ax[0].scatter(r['q0'][3], r['dl2_A_minus_B'], c=col[r['expert_family']], s=12)
        ax[1].scatter(r['q0'][3], r['q0'][5], c=col[r['expert_family']], s=12)
    ax[0].axhline(0, c='k', lw=0.5)
    ax[0].set_xlabel('joint4 inicial [rad]')
    ax[0].set_ylabel('L2(A) - L2(B) [rad]')
    ax[0].set_title('Escenarios var_s/m/l: rama elegida (color) vs coste')
    ax[1].set_xlabel('joint4 inicial [rad]')
    ax[1].set_ylabel('joint6 inicial [rad]')
    ax[1].set_title('Rama elegida según joint4 / joint6 iniciales')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'map_levels.png'), dpi=110)
    plt.close(fig)
    # rama EJECUTADA por posición del cubo: experto vs bc_v1 vs bc_v2w (var_m + var_l)
    eps = [json.loads(x) for x in open(os.path.join(args.out, 'episodes.jsonl'))]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for k, pol in enumerate(('scripted', 'bc_v1', 'bc_v2w')):
        for r in eps:
            if r['policy'] != pol or r['level'] not in ('reach_var_m', 'reach_var_l'):
                continue
            xy = np.asarray(r['obj_xy_cmd'])
            c = col.get(r['executed_final'], '#888888')
            ax[k].scatter(100 * xy[0], 100 * xy[1], c=c, s=14 if r['success'] else 60,
                          marker='o' if r['success'] else 'x', alpha=0.8 if r['success'] else 1.0)
        ax[k].set_title(f'{pol}: rama EJECUTADA (var_m + var_l)\nrojo A, azul B; x = fallo')
        ax[k].set_xlabel('x cubo (mundo) [cm]')
        ax[k].set_ylabel('y cubo (mundo) [cm]')
        ax[k].set_aspect('equal')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'map_executed_family.png'), dpi=110)
    plt.close(fig)
    # trazas joint4 / joint5
    _plot_traces(args, plt)
    _plot_gazebo_vs_kin(args, plt)
    _plot_bc_hist(args, plt)
    print('[plots] listo', args.out)


def _plot_traces(args, plt):
    runs = _load_runs(args.root)
    want = [('kinematic_scripted_reach_var_m', 'experto'), ('kinematic_bc_v1_reach_var_m', 'bc_v1'),
            ('kinematic_bc_v2w_reach_var_m', 'bc_v2w')]
    fig, ax = plt.subplots(2, 3, figsize=(16, 7), sharex=True)
    for k, (name, label) in enumerate(want):
        if name not in runs:
            continue
        d, rows = runs[name]
        for s in rows[:60]:
            arr = _episode_arrays(d, s['seed'])
            q = B.denorm_q(arr['obs'], np.asarray(s['lo']), np.asarray(s['hi']))
            c = '#999999' if s['success'] else '#e07a00'
            lw, al = (0.6, 0.35) if s['success'] else (1.4, 0.9)
            ax[0, k].plot(q[:, 3], c=c, lw=lw, alpha=al)
            ax[1, k].plot(q[:, 4], c=c, lw=lw, alpha=al)
        ax[0, k].set_title(f'{label} · reach_var_m (60 ep., naranja = fallo)')
        ax[0, k].set_ylabel('joint4 [rad]')
        ax[1, k].set_ylabel('joint5 [rad]')
        ax[1, k].set_xlabel('paso')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'traces_j4_j5_var_m.png'), dpi=110)
    plt.close(fig)


def _plot_gazebo_vs_kin(args, plt):
    """Episodios de Gazebo frente al mismo escenario (semilla) en cinemático: joint4/joint5 y distancia."""
    runs = _load_runs(args.root)
    gz = [(n, d, r) for n, (d, rows) in runs.items() if n.startswith('gazebo_') for r in rows]
    if not gz:
        return
    fig, ax = plt.subplots(len(gz), 3, figsize=(15, 2.3 * len(gz)), squeeze=False)
    for i, (name, d, s) in enumerate(gz):
        kin = 'kinematic_' + name[len('gazebo_'):]
        for label, (dd, rows), style in (('Gazebo', (d, [s]), '-'),
                                          ('cinemático', runs.get(kin, (None, [])), '--')):
            row = [r for r in rows if r['seed'] == s['seed']]
            if not row or dd is None:
                continue
            arr = _episode_arrays(dd, s['seed'])
            q = B.denorm_q(arr['obs'], np.asarray(row[0]['lo']), np.asarray(row[0]['hi']))
            ok = 'éxito' if row[0]['success'] else 'FALLO'
            ax[i, 0].plot(q[:, 3], style, label=f'{label} ({ok})')
            ax[i, 1].plot(q[:, 4], style)
            ax[i, 2].plot(1000 * arr['dist'], style)
            if label == 'Gazebo':
                inv = np.nonzero(arr['obs'][:, 16] < 0.5)[0]
                ax[i, 2].scatter(inv, 1000 * arr['dist'][inv], s=6, c='#e07a00', zorder=3)
        ax[i, 0].set_ylabel(f"{name.replace('gazebo_', '')}\n{s['seed']}", fontsize=8)
        ax[i, 0].legend(fontsize=7)
        ax[i, 2].axhline(30, c='k', lw=0.5)
    ax[0, 0].set_title('joint4 [rad]')
    ax[0, 1].set_title('joint5 [rad]')
    ax[0, 2].set_title('distancia al objetivo [mm] (puntos: detección inválida en Gazebo)')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'gazebo_vs_kinematic.png'), dpi=95)
    plt.close(fig)


def _plot_bc_hist(args, plt):
    f = os.path.join(args.out, 'contradiction_arrays.npz')
    if not os.path.exists(f):
        return
    z = np.load(f)
    sat = np.abs(z['a4_expert']) > 0.9
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    bins = np.linspace(-1, 1, 41)
    ax[0].hist(z['a4_expert'][sat], bins=bins, color='#555555')
    ax[0].set_title('Acción joint4 del experto (pasos saturados)')
    for k, name in enumerate(('bc_v1', 'bc_v2w')):
        p = z[f'a4_{name}']
        ax[k + 1].hist(p[sat & ~z['conflict']], bins=bins, alpha=0.6, label='región sin conflicto', density=True)
        ax[k + 1].hist(p[sat & z['conflict']], bins=bins, alpha=0.6, label='región con conflicto', density=True)
        ax[k + 1].set_title(f'{name}: acción joint4 predicha en los estados del experto')
        ax[k + 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'bc_a4_hist.png'), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('ikmap')
    p.add_argument('--out', required=True)
    p.add_argument('--grid', type=int, default=11)
    p.add_argument('--seeds-s', default='915200-915399')
    p.add_argument('--seeds-m', default='915400-915599')
    p.add_argument('--seeds-l', default='915600-915799')
    p = sub.add_parser('rollouts')
    p.add_argument('--policy', choices=['scripted', 'bc_v1', 'bc_v2w'], required=True)
    p.add_argument('--backend', choices=['kinematic', 'gazebo'], default='kinematic')
    p.add_argument('--level', required=True)
    p.add_argument('--seeds', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--tag', default='')
    p = sub.add_parser('analyze')
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p = sub.add_parser('labelrules')
    p.add_argument('--root', required=True)
    p.add_argument('--analysis', required=True)
    p = sub.add_parser('plots')
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    args = ap.parse_args()
    {'ikmap': cmd_ikmap, 'rollouts': cmd_rollouts, 'analyze': cmd_analyze, 'labelrules': cmd_labelrules,
     'plots': cmd_plots}[args.cmd](args)


if __name__ == '__main__':
    main()
