#!/usr/bin/env python3
"""Tablas y figuras del experimento de percepción con dos cámaras (fases A, B y F).

    python3 solverbot_rl/scripts/report_dual_camera.py --root runs/dual_camera \
        --out solverbot_rl/docs/resultados/dual_camera
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _f(v, fmt='{:.1f}'):
    return '—' if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)


def static_tables(root, out):
    p = os.path.join(root, 'audit_static', 'audit_static.json')
    if not os.path.exists(p):
        return []
    r = json.load(open(p))
    L = ['## Fase A — cámara de cabeza (auditoría estática)', '']
    ci = r['camera_info']
    L += ['| cámara | frame | resolución | HFOV (de K) | VFOV | distorsión | frecuencia (tiempo sim.) | antigüedad tras un paso |',
          '|---|---|---|---|---|---|---|---|']
    for k in ('head', 'wrist'):
        c, rt = ci[k], r['rates'][k]
        L.append(f'| {k} | `{c["frame_id"]}` | {c["width"]}×{c["height"]} | {np.degrees(c["hfov_from_k_rad"]):.1f}° | '
                 f'{np.degrees(c["vfov_from_k_rad"]):.1f}° | {c["distortion_model"]} {c["d"]} | {rt["sim_hz"]:.1f} Hz '
                 f'({rt["unique_stamps"]} cuadros) | {r["age_after_step_s"][k]["median"]:.3f} s |')
    rd = r['rates']['depth']
    L += ['', f'Profundidad alineada: {rd["sim_hz"]:.1f} Hz; desfase de sello RGB−profundidad en la pose configurada: '
          f'{r["configured_pose"]["head"]["rgb_depth_dt"]} s.', '']
    h = r['configured_pose']['head']
    L += [f'Pose configurada pan={r["head_cfg"]["pan_joint"]}, tilt={r["head_cfg"]["tilt_joint"]}: cámara en '
          f'{np.round(h["cam_xyz"], 3).tolist()} (base), eje óptico {np.round(h["cam_axis"], 3).tolist()}; '
          f'proyección del cubo real en ({h["u"]:.1f}, {h["v"]:.1f}) px a {h["range_expected"]:.3f} m; profundidad '
          f'medida allí: {h["depth_at_projection"]}; píxeles con profundidad −inf (más cerca que near = 0,105 m): '
          f'{100 * h["frac_depth_neginf"]:.0f} %; estado: **{h["status"]}**.', '']
    sw = r['sweep']
    pans = sorted({s['cmd'][0] for s in sw})
    tilts = sorted({s['cmd'][1] for s in sw})
    L += ['Barrido pan/tilt (cubo en la posición nominal, brazo en home): estado / píxeles de la máscara / % −inf', '',
          '| pan \\ tilt | ' + ' | '.join(f'{t:+.2f}' for t in tilts) + ' |', '|---' * (len(tilts) + 1) + '|']
    for p in pans:
        row = []
        for t in tilts:
            s = next(x for x in sw if x['cmd'] == [p, t])['head']
            row.append(f'{s["status"]} / {s["mask_pixels"]} / {100 * s["frac_depth_neginf"]:.0f}')
        L.append(f'| {p:+.2f} | ' + ' | '.join(row) + ' |')
    g = r['table_grid']
    st = {}
    for s in g:
        st[s['head']['status']] = st.get(s['head']['status'], 0) + 1
    L += ['', f'Mapa de visibilidad de la mesa (cabeza apuntada a cada posición, {len(g)} posiciones): '
          + ', '.join(f'{k}: {v}' for k, v in sorted(st.items()))
          + f'; detecciones: {sum(s["head"]["detected"] for s in g)}.', '']
    L += ['| control positivo (cubo en el aire, vista libre) | estado | píxeles | |proyección − máscara| (px) | '
          'error 3D (mm) |', '|---|---|---|---|---|']
    for s in r['positive_control']:
        h = s['head']
        L.append(f'| {np.round(s["truth"], 2).tolist()} | {h["status"]} | {h["mask_pixels"]} | '
                 f'{_f(h["mask_to_projection_px"], "{:.2f}")} | {_f(1000 * h["err"] if h["err"] else None)} |')
    # figura: mapa de la mesa
    fig, ax = plt.subplots(figsize=(6, 4))
    t = r.get('table_grid')
    xs = [s['truth'][0] for s in t]
    ys = [s['truth'][1] for s in t]
    col = ['tab:green' if s['head']['detected'] else 'tab:red' for s in t]
    ax.scatter(xs, ys, c=col, s=60, marker='s')
    ax.add_patch(plt.Rectangle((0.22 - 0.07, 0.56 - 0.07), 0.14, 0.14, fill=False, ls='--', label='reach_var_l (±7 cm)'))
    ax.plot([0.22], [0.56], 'kx', label='posición nominal')
    ax.set_xlabel('x base (m)')
    ax.set_ylabel('y base (m)')
    ax.set_title('Cabeza apuntada a cada posición: rojo = no ve el cubo')
    ax.legend(fontsize=7)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'head_table_visibility.png'), dpi=110)
    plt.close(fig)
    # hoja de contacto
    names = ['cfg_pose_head', 'cfg_pose_depth', 'cfg_pose_wrist', 'sweep_p+1.20_t+0.20_head',
             'sweep_p+1.20_t+0.20_depth', 'sweep_p+0.60_t+0.40_head', 'grid_x+0.19_y+0.81_head',
             'grid_x+0.19_y+0.81_depth', 'floor_x+0.90_y+0.00_head', 'positive_0.55_+0.00_1.38_head',
             'positive_0.55_+0.00_1.38_depth', 'positive_0.65_+0.25_1.30_head']
    titles = ['cabeza, pose de la tarea (pan 1,20 / tilt 0,78)', 'profundidad (magenta = −inf, placa)',
              'muñeca, home', 'cabeza pan 1,20 tilt 0,20 (brazo)', 'profundidad', 'cabeza pan 0,60 tilt 0,40',
              'cabeza → cubo en y=0,81 (borde lejano)', 'profundidad', 'cabeza → piso adelante (frente del robot)',
              'control +: cubo en el aire', 'profundidad', 'control +']
    fig, axs = plt.subplots(3, 4, figsize=(16, 9.6))
    for a, n, ti in zip(axs.flat, names, titles):
        f = os.path.join(root, 'audit_static', n + '.png')
        if os.path.exists(f):
            a.imshow(plt.imread(f))
        a.set_title(ti, fontsize=9)
        a.axis('off')
    fig.suptitle('Cruz verde: proyección de la pose REAL del cubo (solo diagnóstico)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'head_audit_sheet.png'), dpi=80)
    plt.close(fig)
    return L


def coverage(root, out):
    L = ['## Fase B — cobertura por cámara (misma trayectoria, registro independiente)', '',
         '| corrida | episodios | pasos | muñeca válida | cabeza detecta | solo muñeca | solo cabeza | ambas | ninguna | '
         'err. muñeca (mm) | err. muñeca con FK en el instante de la imagen (mm) | antigüedad imagen (s) | '
         'estado cabeza (pasos) |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    per_seed = []
    for d in sorted(glob.glob(os.path.join(root, 'audit_ep', '*'))):
        files = sorted(glob.glob(os.path.join(d, 'cams_*.json')))
        if not files:
            continue
        allst = []
        for f in files:
            st = json.load(open(f))[1:]                      # [0] = después del reset (antes de actuar)
            allst += st
            w = np.array([s['obs_valid'] for s in st])
            h = np.array([s['head']['detected'] for s in st])
            per_seed.append({'run': os.path.basename(d), 'seed': int(f.split('_')[-1][:-5]), 'steps': len(st),
                             'wrist': float(w.mean()), 'head': float(h.mean()), 'none': float((~w & ~h).mean())})
        w = np.array([s['obs_valid'] for s in allst])
        h = np.array([s['head']['detected'] for s in allst])
        we = [s['wrist']['err'] for s in allst if s['wrist']['err'] is not None]
        wi = [s['wrist']['err_at_image_time'] for s in allst if s['wrist'].get('err_at_image_time') is not None]
        ages = [s['head']['age'] for s in allst] + [s['wrist']['age'] for s in allst]
        hs = {}
        for s in allst:
            hs[s['head']['status']] = hs.get(s['head']['status'], 0) + 1
        n = len(allst)
        L.append(f'| {os.path.basename(d)} | {len(files)} | {n} | {100 * w.mean():.1f} % | {100 * h.mean():.1f} % | '
                 f'{100 * (w & ~h).mean():.1f} % | {100 * (h & ~w).mean():.1f} % | {100 * (w & h).mean():.1f} % | '
                 f'{100 * (~w & ~h).mean():.1f} % | {_f(np.mean(we) * 1000 if we else None)} (máx '
                 f'{_f(np.max(we) * 1000 if we else None)}) | {_f(np.mean(wi) * 1000 if wi else None)} (máx '
                 f'{_f(np.max(wi) * 1000 if wi else None)}) | {np.median(ages):.3f} (máx {np.max(ages):.3f}) | '
                 + ', '.join(f'{k} {v}' for k, v in sorted(hs.items())) + ' |')
    L += ['', '| corrida | semilla | pasos | muñeca válida | cabeza detecta | ninguna |', '|---|---|---|---|---|---|']
    for r in per_seed:
        L.append(f'| {r["run"]} | {r["seed"]} | {r["steps"]} | {100 * r["wrist"]:.0f} % | {100 * r["head"]:.0f} % | '
                 f'{100 * r["none"]:.0f} % |')
    return L


def frames_sheet(root, out, run='bc_v2w_var_m', seeds=(902002, 902003), steps=(0, 6, 10, 12, 14, 20, 30, 40)):
    for seed in seeds:
        f = os.path.join(root, 'audit_ep', run, f'frames_{seed}.npz')
        if not os.path.exists(f):
            continue
        z = np.load(f)
        cams = json.load(open(os.path.join(root, 'audit_ep', run, f'cams_{seed}.json')))
        idx = [i for i, s in enumerate(z['step']) if s in steps]
        fig, axs = plt.subplots(3, len(idx), figsize=(2.2 * len(idx), 6.4))
        for c, i in enumerate(idx):
            s = int(z['step'][i])
            st = cams[s]
            axs[0, c].imshow(z['wrist'][i])
            axs[0, c].set_title(f'paso {s}: muñeca {"✓" if st["obs_valid"] else "✗"}\n'
                                f'{st["wrist"]["status"]}', fontsize=8)
            axs[1, c].imshow(z['head'][i])
            axs[1, c].set_title(f'cabeza: {st["head"]["status"]}', fontsize=8)
            d = z['depth'][i].astype(float)
            vis = np.where(np.isneginf(d), -0.5, np.where(np.isfinite(d), d, 3.0))
            axs[2, c].imshow(vis, cmap='magma_r', vmin=-0.5, vmax=3.0)
            axs[2, c].set_title(f'prof.: −inf (claro) {100 * np.mean(np.isneginf(d)):.0f} %', fontsize=8)
            for a in axs[:, c]:
                a.axis('off')
        fig.suptitle(f'bc_v2w, reach_var_m, semilla {seed}: las dos cámaras en la misma trayectoria', fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f'frames_{seed}.png'), dpi=80)
        plt.close(fig)


def counterfactuals(root, out):
    L = ['## Fase F — contrafácticos de la observación del objeto (misma política congelada, mismas semillas)', '',
         '| política / nivel | modo | privilegiado | éxitos | IC 95 % | dmin medio (mm) | dfinal medio (mm) | '
         'TCP z final | pasos | bandera válida vista por la política | error de la estimación vista (mm) | fallos |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    groups = sorted(glob.glob(os.path.join(root, 'cf', '*')))
    for g in groups:
        for mode in ('perception', 'hold_valid', 'first_estimate', 'oracle'):
            p = os.path.join(g, mode, 'summary.json')
            if not os.path.exists(p):
                continue
            s = json.load(open(p))
            rows = [json.loads(x) for x in open(os.path.join(g, mode, 'index.jsonl'))]
            pv = np.mean([r['policy_valid_rate'] for r in rows])
            L.append(f'| {os.path.basename(g)} | {mode} | {"sí" if s["privileged_obs"] else "no"} | '
                     f'{s["successes"]}/{s["episodes"]} | [{100 * s["success_wilson95"][0]:.0f} %, '
                     f'{100 * s["success_wilson95"][1]:.0f} %] | {s["dmin_mm"]["mean"]:.1f} | {s["dfinal_mm"]["mean"]:.1f} | '
                     f'{s["tcp_z2_final"]["mean"]:+.3f} | {s["length"]["mean"]:.1f} | {100 * pv:.0f} % | '
                     f'{s["policy_est_err_mean_mm"]:.1f} | {s["failed_seeds"]} |')
    return L


def mcnemar_exact(b, c):
    """p bilateral exacto de McNemar con b, c pares discordantes."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(root, ref='perception'):
    L = ['Comparación PAREADA contra `perception` (mismas semillas): b = falla con percepción y tiene éxito con el '
         'modo, c = al revés; McNemar exacto bilateral.', '',
         '| grupo | modo | semillas comunes | b (recupera) | c (pierde) | p |', '|---|---|---|---|---|---|']
    for g in sorted(glob.glob(os.path.join(root, 'cf', '*'))):
        rp = os.path.join(g, ref, 'index.jsonl')
        if not os.path.exists(rp):
            continue
        base = {r['seed']: r['success'] for r in map(json.loads, open(rp))}
        for mode in ('hold_valid', 'first_estimate', 'oracle'):
            p = os.path.join(g, mode, 'index.jsonl')
            if not os.path.exists(p):
                continue
            other = {r['seed']: r['success'] for r in map(json.loads, open(p))}
            common = sorted(set(base) & set(other))
            b = sum((not base[s]) and other[s] for s in common)
            c = sum(base[s] and not other[s] for s in common)
            L.append(f'| {os.path.basename(g)} | {mode} | {len(common)} | {b} | {c} | {mcnemar_exact(b, c):.3f} |')
    return L


def reproducibility(root):
    L = ['## Reproducibilidad por semilla entre lanzamientos de la simulación', '',
         '| lanzamiento | modo | antigüedad mediana de la imagen de muñeca (s) | ' + 'resultado por semilla (éxito, dfinal mm, pasos) |',
         '|---|---|---|---|']
    for d in sorted(glob.glob(os.path.join(root, 'repro', '*'))):
        rows = [json.loads(x) for x in open(os.path.join(d, 'index.jsonl'))]
        name = os.path.basename(d)
        L.append(f'| {name.split("_")[-1]} | {name.rsplit("_", 1)[0]} | '
                 f'{_f(rows[0].get("wrist_image_age_median_s"), "{:.3f}")} | '
                 + '; '.join(f'{r["seed"]}: {"✓" if r["success"] else "✗"} {r["dfinal_mm"]:.0f} / {r["length"]}'
                             for r in rows) + ' |')
    return L


def summary_rows(title, pattern, root):
    L = [title, '', '| corrida | éxitos | IC 95 % | dmin medio | dfinal medio | TCP z final | fallos |',
         '|---|---|---|---|---|---|---|']
    for p in sorted(glob.glob(os.path.join(root, pattern, 'summary.json'))):
        s = json.load(open(p))
        L.append(f'| {os.path.basename(os.path.dirname(p))} | {s["successes"]}/{s["episodes"]} | '
                 f'[{100 * s["success_wilson95"][0]:.0f} %, {100 * s["success_wilson95"][1]:.0f} %] | '
                 f'{s["dmin_mm"]["mean"]:.1f} | {s["dfinal_mm"]["mean"]:.1f} | {s["tcp_z2_final"]["mean"]:+.3f} | '
                 f'{s["failed_seeds"] if len(s["failed_seeds"]) <= 8 else str(len(s["failed_seeds"])) + " semillas"} |')
    return L


def trace_plot(root, out, group='bc_v2w_var_m', seed=902003):
    from solverbot_rl.config import load_config, load_urdf
    from solverbot_rl.kinematics import UrdfChain
    ch, cfg = UrdfChain(load_urdf()), load_config()
    lo, hi = ch.limits(cfg['robot']['arm_joints'])
    fig, axs = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    for mode, c in (('perception', 'tab:blue'), ('hold_valid', 'tab:purple'), ('first_estimate', 'tab:orange'),
                    ('oracle', 'tab:green')):
        f = os.path.join(root, 'cf', group, mode, 'episodes', f'ep_{seed}.npz')
        if not os.path.exists(f):
            continue
        z = np.load(f)
        q = (z['obs'][:, :6] + 1) / 2 * (hi - lo) + lo
        t = np.arange(len(z['dist']))
        axs[0].plot(t, 1000 * z['dist'], c=c, label=mode)
        axs[1].plot(t, z['actions'][:, 3], c=c)
        axs[2].plot(t, q[:, 3], c=c)
        axs[3].plot(t, z['tcp_z2'], c=c)
        inv = ~z['obj_valid'].astype(bool)
        axs[0].scatter(t[inv], 1000 * z['dist'][inv], c=c, s=6, marker='x')
    axs[0].axhline(30, c='k', ls=':', lw=1)
    axs[0].set_ylabel('dist. TCP–objetivo (mm)\n(x = muñeca sin detección)')
    axs[1].set_ylabel('acción joint4')
    axs[2].set_ylabel('q joint4 (rad)')
    axs[3].set_ylabel('z del eje del TCP\n(−1 = vertical)')
    axs[3].set_xlabel('paso')
    axs[0].legend(fontsize=8)
    fig.suptitle(f'{group}, semilla {seed}: misma política, distinta información del objeto', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'trace_{group}_{seed}.png'), dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='runs/dual_camera')
    ap.add_argument('--out', required=True)
    ap.add_argument('--trace-seeds', default='902003')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    L = static_tables(args.root, args.out) + [''] + coverage(args.root, args.out) + [''] + \
        counterfactuals(args.root, args.out) + [''] + paired(args.root) + [''] + reproducibility(args.root) + [''] + \
        summary_rows('## Backend cinemático (detector simulado continuo, sin campo visual ni física)', 'kin/*',
                     args.root) + [''] + \
        summary_rows('## Guardián de compromiso de rama (cinemático, mismas semillas base / guardián)', 'guard/*',
                     args.root)
    frames_sheet(args.root, args.out)
    for s in args.trace_seeds.split(','):
        trace_plot(args.root, args.out, seed=int(s))
    tables = [('runs/dual_camera/branch_analysis/branch_table.md', 'branch_table_eval_previas.md'),
              ('runs/dual_camera/branch_cf/branch_table.md', 'branch_table_contrafacticos.md')]
    for src, dst in tables:
        if os.path.exists(src):
            with open(src) as fi, open(os.path.join(args.out, dst), 'w') as fo:
                fo.write(fi.read())
    with open(os.path.join(args.out, 'tables.md'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
