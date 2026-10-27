#!/usr/bin/env python3
"""Captura una imagen de la cámara de cabeza y una de la cámara de la pinza (PNG) desde una simulación en marcha.
Útil para verificar la arena sin GUI (p. ej. en WSL sin GPU o por SSH).

    python3 solverbot_rl/scripts/capture_cameras.py --out capturas/ --pan 1.0 --tilt 0.35
"""
import argparse
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_sim import write_png  # noqa: E402

HOME = [1.753, 0.046, -0.671, -0.626, 0.918, 1.128]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='capturas')
    ap.add_argument('--pan', type=float, default=0.0)
    ap.add_argument('--tilt', type=float, default=0.3)
    ap.add_argument('--seconds', type=float, default=20.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rclpy.init()
    n = Node('capture_cameras')
    imgs = {}

    def cb(key, m):
        imgs[key] = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[..., :3]
    n.create_subscription(Image, '/camera/camera/color/image_raw', lambda m: cb('head', m), 5)
    n.create_subscription(Image, '/wrist_camera/image_raw', lambda m: cb('wrist', m), 5)
    ph = n.create_publisher(Float64MultiArray, '/head_position_controller/commands', 10)
    pa = n.create_publisher(Float64MultiArray, '/arm_position_controller/commands', 10)
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        ph.publish(Float64MultiArray(data=[args.pan, args.tilt]))
        pa.publish(Float64MultiArray(data=HOME))
        rclpy.spin_once(n, timeout_sec=0.1)
    for k, im in imgs.items():
        write_png(os.path.join(args.out, f'{k}.png'), im)
    print('capturadas:', {k: v.shape for k, v in imgs.items()})
    n.destroy_node()
    rclpy.try_shutdown()
    return 0 if len(imgs) == 2 else 1


if __name__ == '__main__':
    sys.exit(main())
