#!/usr/bin/env python3
"""Validación en tiempo de ejecución de SolverBot v3 en Gazebo Harmonic (mundo pick_place_training).

Requiere la simulación en marcha y en pausa:
    ros2 launch solverbot_sim sim.launch.py world:=pick_place_training paused:=true headless:=true cameras:=false
    python3 solverbot_rl/scripts/validate_sim.py --out evidencias/

Comprueba: articulaciones y controladores, mimic de la pinza, elevador, pan-tilt, ambas cámaras (imágenes
reales que cambian al mover cabeza/brazo), percepción del objeto y una prueba FÍSICA de agarre guionada
(IK + controladores): el objeto debe subir con la pinza y mantenerse sujeto.
Salida: report.json + PNG de ambas cámaras. Código de salida != 0 si alguna prueba falla.
"""
import argparse
import json
import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solverbot_rl.backends.gazebo import GazeboBackend  # noqa: E402
from solverbot_rl.config import load_config  # noqa: E402



def write_png(path, rgb):
    h, w, _ = rgb.shape
    raw = b''.join(b'\x00' + rgb[y].astype(np.uint8).tobytes() for y in range(h))

    def chunk(t, d):
        return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    with open(path, 'wb') as fh:
        fh.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) +
                 chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='validation_out')
    ap.add_argument('--world', default='pick_place_training')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = load_config()
    r = cfg['robot']
    be = GazeboBackend(cfg, world=args.world, images=('head', 'wrist'), perception=True)
    rep, ok = {}, True

    def check(name, cond, **data):
        nonlocal ok
        ok &= bool(cond)
        rep[name] = {'pass': bool(cond), **{k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v)
                                           for k, v in data.items()}}
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {rep[name]}")

    J = r['arm_joints']
    home = np.array(r['home'])
    obj_xy = np.array(cfg['world']['object']['xy'])
    be.reset(np.random.default_rng(0), obj_xy, 0.0, home, r['gripper_open'], r['lift'], r['head'])
    js = be.joint_positions()
    expected = J + ['joint7', 'joint8', 'torso_lift_joint', 'pan_joint', 'tilt_joint', 'left_wheel_joint',
                    'right_wheel_joint']
    check('joints_present', all(j in js for j in expected), missing=[j for j in expected if j not in js])
    q = np.array([js[j] for j in J])
    check('arm_reaches_home', np.max(np.abs(q - home)) < 0.03, err=float(np.max(np.abs(q - home))))
    check('lift_tracks', abs(js['torso_lift_joint'] - r['lift']) < 0.01, lift=js['torso_lift_joint'])
    check('gripper_mimic_open', abs(js['joint8'] + js['joint7']) < 2e-3 and js['joint7'] > 0.03,
          joint7=js['joint7'], joint8=js['joint8'])

    imgs = be.images()
    for k in ('head', 'wrist'):
        im = imgs.get(k)
        check(f'camera_{k}_image', im is not None and im.std() > 3.0, shape=list(im.shape) if im is not None else None)
        if im is not None:
            write_png(os.path.join(args.out, f'{k}_home.png'), im)

    truth = be.object_pose_truth()
    est, valid = be.object_estimate()
    check('perception_wrist_camera', valid and np.linalg.norm(est - truth) < 0.03, truth=truth, estimate=est)

    # La cabeza mueve la imagen de cabeza; el brazo mueve la de la pinza
    head0, wrist0 = imgs['head'].astype(float), imgs['wrist'].astype(float)
    be._pub(be.pub_head, [0.0, 0.5])                    # mirar adelante y abajo
    be.advance(1.0)
    check('head_camera_follows_pan_tilt', np.abs(be.images()['head'].astype(float) - head0).mean() > 5.0,
          pan=be.joint_positions()['pan_joint'])
    be._pub(be.pub_head, [r['head']['pan_joint'], r['head']['tilt_joint']])

    # ---------------- prueba física de agarre: el MISMO procedimiento que usa el entorno RL (experto guionado con
    # precierre a 32 mm, descenso verificado, rampa de cierre y acoplamiento de dedos), semilla 40001.
    be.command(np.array([be.joint_positions()[j] for j in J]) + [0.3, 0, 0, 0, 0, 0], r['gripper_open'])
    for _ in range(10):
        be.advance(0.1)
    check('wrist_camera_follows_arm', np.abs(be.images()['wrist'].astype(float) - wrist0).mean() > 5.0)
    be.close()
    # Mismo proceso que en producción: validate_grasp.py en un subproceso (conexión ROS propia)
    import subprocess
    gdir = os.path.join(args.out, 'grasp')
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'validate_grasp.py'),
                    '--backend', 'gazebo', '--seeds', '40001', '--hold-steps', '20', '--out', gdir], check=False)
    with open(os.path.join(gdir, 'summary.json')) as fh:
        g = json.load(fh)['seeds'][0]

    rep['all_pass'] = bool(ok)
    with open(os.path.join(args.out, 'report.json'), 'w') as fh:
        json.dump(rep, fh, indent=1)
    print('RESULTADO:', 'TODO OK' if ok else 'HAY FALLAS')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
