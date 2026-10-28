#!/usr/bin/env python3
"""Auditoría de las dos cámaras en Gazebo (diagnóstico; la pose real del cubo es SOLO referencia de medición).

    # Fase A: CameraInfo, frecuencias, barrido pan/tilt, mapa de visibilidad de la mesa, control positivo
    python3 solverbot_rl/scripts/audit_cameras.py static --out runs/dual_camera/audit_static

    # Fase B: misma política y semillas, registro independiente de ambas cámaras en cada paso
    python3 solverbot_rl/scripts/audit_cameras.py episodes --policy solverbot_rl/docs/resultados/reach_bc_v1/modelos/bc_v2w.zip \
        --level reach_var_m --seeds 902002-902003 --out runs/dual_camera/audit_bc_v2w_var_m

La política (episodes) recibe exactamente la observación de 31 valores del entorno: la sonda solo se suscribe a
tópicos y no modifica el entorno. La pose del cabezal no forma parte de la observación.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solverbot_rl.camera_probe import CameraProbe, look_at  # noqa: E402
from solverbot_rl.envs.pick_place import SolverBotPickPlaceEnv  # noqa: E402
from solverbot_rl.imitation import dataset as D  # noqa: E402
from solverbot_rl.imitation.experiment import env_kwargs, level_index, load_experiment, parse_seeds  # noqa: E402
from solverbot_rl.imitation.rollout import run_episode  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_sim import write_png  # noqa: E402


def depth_vis(d, lo=0.1, hi=2.5):
    """Profundidad -> RGB: gris (cerca claro), magenta = -inf (más cerca que near), negro = +inf / sin dato."""
    d = np.asarray(d, float)
    out = np.zeros(d.shape + (3,), np.uint8)
    fin = np.isfinite(d)
    g = (255 * (1 - np.clip((d[fin] - lo) / (hi - lo), 0, 1))).astype(np.uint8)
    out[fin] = np.stack([g, g, g], 1)
    out[np.isneginf(d)] = (255, 0, 255)
    return out


def mark(rgb, u, v, color=(0, 255, 0), r=4):
    """Cruz en la proyección de la pose REAL (solo para las imágenes de evidencia)."""
    img = rgb.copy()
    h, w = img.shape[:2]
    if not (np.isfinite(u) and np.isfinite(v)):
        return img
    u, v = int(round(u)), int(round(v))
    for k in range(-r, r + 1):
        for x, y in ((u + k, v), (u, v + k)):
            if 0 <= x < w and 0 <= y < h:
                img[y, x] = color
    return img


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return str(x)
    return x


def save_views(out, tag, probe, meas):
    img = probe.img
    h = meas.get('head', {})
    if 'head' in img:
        write_png(os.path.join(out, f'{tag}_head.png'), mark(img['head'][1], h.get('u', np.nan), h.get('v', np.nan)))
    if 'depth' in img:
        write_png(os.path.join(out, f'{tag}_depth.png'), depth_vis(img['depth'][1]))
    if 'wrist' in img:
        w = meas.get('wrist', {})
        write_png(os.path.join(out, f'{tag}_wrist.png'), mark(img['wrist'][1], w.get('u', np.nan), w.get('v', np.nan)))


def settle_head(be, probe, pan, tilt, seconds=0.8):
    be._pub(be.pub_head, [pan, tilt])
    for _ in range(int(round(seconds / 0.1))):
        be.advance(0.1)
    probe.wait_fresh()


def info_dict(m):
    return {'frame_id': m.header.frame_id, 'width': m.width, 'height': m.height, 'distortion_model': m.distortion_model,
            'd': list(m.d), 'k': list(m.k), 'p': list(m.p),
            'hfov_from_k_rad': float(2 * np.arctan(m.width / 2 / m.k[0])),
            'vfov_from_k_rad': float(2 * np.arctan(m.height / 2 / m.k[4]))}


def static(args, env):
    be = env.unwrapped.backend
    probe = CameraProbe(be)
    env.reset(seed=args.seed, options={'level': level_index(env, 'reach')})
    probe.wait_ready()
    rep = {'camera_info': {k: info_dict(m) for k, m in probe.info.items()}, 'head_cfg': be.cfg['robot']['head']}
    # Frecuencias y antigüedad con el mundo avanzando a pasos de 0.1 s (como el entorno)
    ages = []
    for _ in range(args.rate_steps):
        be.advance(0.1)
        ages.append(probe.wait_fresh())
    rep['rates'] = probe.rates()
    rep['age_after_step_s'] = {k: {'median': float(np.median([a[k] for a in ages])),
                                    'max': float(np.max([a[k] for a in ages]))} for k in ages[0]}
    js = be.joint_positions()
    m = probe.measure()
    rep['configured_pose'] = m
    save_views(args.out, 'cfg_pose', probe, m)
    rep['tf_head_optical_in_base'] = be.chain.link_pose('camera_color_optical_frame', js).tolist()
    rep['tf_wrist_optical_in_base'] = be.chain.link_pose('wrist_sim_camera_optical_frame', js).tolist()
    obj0 = be.object_pose_truth()

    # 1. Barrido pan/tilt con el cubo en su posición nominal y el brazo en home
    sweep = []
    pans = [float(v) for v in args.pans.split(',')]
    tilts = [float(v) for v in args.tilts.split(',')]
    for pan in pans:
        for tilt in tilts:
            settle_head(be, probe, pan, tilt)
            m = probe.measure()
            m['cmd'] = [pan, tilt]
            sweep.append(m)
            save_views(args.out, f'sweep_p{pan:+.2f}_t{tilt:+.2f}', probe, m)
            h = m['head']
            print(f'[sweep] pan={pan:+.2f} tilt={tilt:+.2f} estado={h["status"]} mascara={h["mask_pixels"]} '
                  f'-inf={h["frac_depth_neginf"]:.2f} <0.3m={h["frac_depth_lt_0_30"]:.2f} '
                  f'det={h["detected"]} err={h["err"]}', flush=True)
    rep['sweep'] = sweep
    # 2. Pose "mirar al cubo" (mejor caso geométrico dentro de los límites del pan/tilt)
    pan, tilt = look_at(be.chain, be.joint_positions(), obj0)
    settle_head(be, probe, pan, tilt)
    m = probe.measure()
    m['cmd'] = [pan, tilt]
    rep['look_at_nominal'] = m
    save_views(args.out, 'lookat_nominal', probe, m)
    print(f'[lookat] nominal pan={pan:+.3f} tilt={tilt:+.3f} -> {m["head"]["status"]}', flush=True)

    # 3. Mapa de visibilidad: cubo teletransportado sobre toda la mesa, cabeza apuntando a él
    t = be.cfg['world']['table']
    xs = np.arange(t['xy'][0] - t['size'][0] / 2 + 0.04, t['xy'][0] + t['size'][0] / 2 - 0.03, args.grid)
    ys = np.arange(t['xy'][1] - t['size'][1] / 2 + 0.04, t['xy'][1] + t['size'][1] / 2 - 0.03, args.grid)
    z = be.table_h + be.obj_half_h + 0.003
    grid = []
    for x in xs:
        for y in ys:
            be.set_pose_in_base(be.obj_name, [x, y, z], 0.0)
            truth = np.array([x, y, z - 0.003])
            pan, tilt = look_at(be.chain, be.joint_positions(), truth)
            settle_head(be, probe, pan, tilt, seconds=0.6)
            m = probe.measure()
            m['cmd'] = [pan, tilt]
            grid.append(m)
            save_views(args.out, f'grid_x{x:+.2f}_y{y:+.2f}', probe, m)
            h = m['head']
            print(f'[grid] x={x:+.2f} y={y:+.2f} pan={pan:+.2f} tilt={tilt:+.2f} estado={h["status"]} '
                  f'mascara={h["mask_pixels"]} det={h["detected"]} err={h["err"]}', flush=True)
    rep['table_grid'] = grid
    # 4. Cubo en el piso delante del robot: ¿qué tapa la vista hacia adelante y abajo?
    floor = []
    for x, y in ((0.9, 0.0), (1.3, 0.0)):
        be.set_pose_in_base(be.obj_name, [x, y, be.obj_half_h + 0.002], 0.0)
        be.advance(0.3)
        truth = be.object_pose_truth()
        pan, tilt = look_at(be.chain, be.joint_positions(), truth)
        settle_head(be, probe, pan, tilt)
        m = probe.measure()
        m['cmd'] = [pan, tilt]
        floor.append(m)
        save_views(args.out, f'floor_x{x:+.2f}_y{y:+.2f}', probe, m)
        h = m['head']
        print(f'[piso] x={x} y={y} estado={h["status"]} mascara={h["mask_pixels"]} det={h["detected"]}', flush=True)
    rep['floor_front'] = floor
    # 5. Control positivo: cubo en el aire con línea de vista libre (0.5-0.8 m) -> valida RGB-D + TF + CameraInfo.
    #    Cada cuadro se renderiza al inicio del paso de 0.1 s (antigüedad medida 0.09 s): se usa la pose ORDENADA
    #    (caída en ~10 ms < 1 mm) y además se registra la pose real al final del paso.
    pos = []
    for p in ((0.55, 0.0, 1.38), (0.65, 0.25, 1.30), (0.75, -0.25, 1.45), (0.50, 0.10, 1.25)):
        pan, tilt = look_at(be.chain, be.joint_positions(), p)
        settle_head(be, probe, pan, tilt)
        be.set_pose_in_base(be.obj_name, list(p), 0.0)
        be.advance(0.1)
        probe.wait_fresh()
        m = probe.measure(truth=np.asarray(p, float), yaw=0.0)
        m['cmd'] = [pan, tilt]
        m['truth_end_of_step'] = be.object_pose_truth().tolist()
        pos.append(m)
        save_views(args.out, f'positive_{p[0]:.2f}_{p[1]:+.2f}_{p[2]:.2f}', probe, m)
        h = m['head']
        print(f'[control+] p={p} estado={h["status"]} mascara={h["mask_pixels"]} det={h["detected"]} '
              f'err={h["err"]} proy-mascara={h["mask_to_projection_px"]}', flush=True)
    rep['positive_control'] = pos
    with open(os.path.join(args.out, 'audit_static.json'), 'w') as fh:
        json.dump(jsonable(rep), fh, indent=1)
    print('[audit] listo:', os.path.join(args.out, 'audit_static.json'), flush=True)


def episodes(args, env):
    from solverbot_rl.imitation.bc import load_policy
    be = env.unwrapped.backend
    probe = CameraProbe(be)
    act = load_policy(args.policy, deterministic=True)
    lvl = level_index(env, args.level)
    os.makedirs(args.out, exist_ok=True)
    done = {r['seed'] for r in D.read_index(args.out)}
    for seed in [s for s in parse_seeds(args.seeds) if s not in done]:
        steps, frames = [], {'head': [], 'depth': [], 'wrist': [], 'step': []}

        def hook(e, obs, a):
            probe.wait_ready()
            probe.wait_fresh()
            js = be.joint_positions()
            m = probe.measure(js=js)
            m.update({'step': len(steps), 'obs_valid': bool(obs[16] > 0.5),
                      'obs_est': (obs[13:16] + obs[7:10]).tolist(), 'action': None if a is None else a.tolist(),
                      'q': [js[j] for j in e.unwrapped.arm_joints],
                      'tcp_z': e.unwrapped.chain.link_pose('tcp_link', js)[:3, 2].tolist()})
            steps.append(m)
            if args.image_every and (len(steps) - 1) % args.image_every == 0:
                frames['step'].append(len(steps) - 1)
                for k in ('head', 'depth', 'wrist'):
                    frames[k].append(probe.img[k][1].copy())

        arrays, s = run_episode(env, seed, lvl, act=act, rng=np.random.default_rng(seed), on_step=hook)
        s.update({'policy': args.policy, 'expert_in_loop': False, 'head_pose': be.cfg['robot']['head']})
        D.save_episode(args.out, arrays, s)
        with open(os.path.join(args.out, f'cams_{seed}.json'), 'w') as fh:
            json.dump(jsonable(steps), fh)
        if frames['step']:
            np.savez_compressed(os.path.join(args.out, f'frames_{seed}.npz'),
                                **{k: np.asarray(v) for k, v in frames.items()})
        hv = np.mean([st['head']['detected'] for st in steps])
        wv = np.mean([st['obs_valid'] for st in steps])
        print(f'[episodes] seed={seed} éxito={s["success"]} pasos={s["length"]} dmin={s["dmin_mm"]:.1f} '
              f'dfin={s["dfinal_mm"]:.1f} muñeca_válida={wv:.2f} cabeza_detecta={hv:.2f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['static', 'episodes'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--experiment', default='reach_bc_v1')
    ap.add_argument('--seed', type=int, default=902000)
    ap.add_argument('--rate-steps', type=int, default=40)
    ap.add_argument('--pans', default='0.0,0.6,0.9,1.2,1.4,1.57')
    ap.add_argument('--tilts', default='-0.3,0.0,0.2,0.4,0.6,0.785')
    ap.add_argument('--grid', type=float, default=0.10, help='paso (m) del mapa de visibilidad de la mesa')
    ap.add_argument('--policy')
    ap.add_argument('--level', default='reach_var_m')
    ap.add_argument('--seeds', default='902002-902003')
    ap.add_argument('--head', default=None, help='pan,tilt fijos del cabezal (por defecto los de la configuración)')
    ap.add_argument('--image-every', type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    kw = env_kwargs(load_experiment(args.experiment))
    if args.head:
        pan, tilt = (float(v) for v in args.head.split(','))
        kw['overrides']['robot.head'] = {'pan_joint': pan, 'tilt_joint': tilt}
    env = SolverBotPickPlaceEnv(backend='gazebo', **kw)
    try:
        (static if args.mode == 'static' else episodes)(args, env)
    finally:
        env.close()


if __name__ == '__main__':
    main()
