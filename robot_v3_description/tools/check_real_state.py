#!/usr/bin/env python3
"""Prueba de real_state.launch.py SIN el robot: imita los /joint_states de la Jetson en un dominio de prueba.

Publica en /joint_states lo mismo que piper_arm_controller.py (joint1..6 en rad, joint7 = +a/2, joint8 = -a/2)
y, por separado, pan_joint/tilt_joint (como pan_tilt_node). Verifica que:
  * /robot_v3/joint_states contenga las 13 articulaciones (elevador y ruedas completadas en 0);
  * la TF de tcp_link, link7/link8 y camera_link coincida con la cinemática del URDF para esos valores.
Se niega a correr en ROS_DOMAIN_ID=10 (allí publicaría estados falsos junto a los reales).

    export ROS_DOMAIN_ID=42
    ros2 launch robot_v3_description real_state.launch.py &
    ros2 run robot_v3_description check_real_state.py
"""
import math
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtime_checks import quat_to_matrix  # noqa: E402
from validate_v3 import DEFAULT_HOME_DEG, Urdf  # noqa: E402

GRIPPER_MM = 69.4                       # apertura inicial que usa piper_arm_controller.py
PAN_TILT = {'pan_joint': 0.4, 'tilt_joint': 0.2}


def main():
    if os.environ.get('ROS_DOMAIN_ID', '0') == '10':
        sys.exit('ROS_DOMAIN_ID=10 es el dominio del robot real: este script publica estados FALSOS.')
    rclpy.init()
    node = Node('robot_v3_fake_jetson')
    pub = node.create_publisher(JointState, '/joint_states', 10)
    merged, urdf = {}, {}
    node.create_subscription(JointState, '/robot_v3/joint_states', lambda m: merged.update(zip(m.name, m.position)), 10)
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(String, '/robot_description', lambda m: urdf.setdefault('u', Urdf(m.data)), qos)
    buf = Buffer()
    TransformListener(buf, node, spin_thread=False)

    arm = [math.radians(v) for v in DEFAULT_HOME_DEG]
    half = min(0.035, GRIPPER_MM / 1000.0 / 2.0)
    expected = dict(zip([f'joint{i}' for i in range(1, 9)], arm + [half, -half]))
    expected.update(PAN_TILT)

    def publish():
        now = node.get_clock().now().to_msg()
        js = JointState(name=[f'joint{i}' for i in range(1, 9)], position=arm + [half, -half])
        js.header.stamp = now
        pub.publish(js)                                     # mensaje "Jetson brazo"
        ph = JointState(name=list(PAN_TILT), position=list(PAN_TILT.values()))
        ph.header.stamp = now
        pub.publish(ph)                                     # mensaje "pan_tilt_node"

    results = []

    def check(name, ok, detail=''):
        results.append(ok)
        print(f"[{'OK' if ok else 'FALLA'}] {name}" + (f' — {detail}' if detail else ''), flush=True)

    end = time.monotonic() + 90   # joint_state_publisher espera /robot_description antes de publicar
    while time.monotonic() < end:
        publish()
        rclpy.spin_once(node, timeout_sec=0.05)
        if 'u' in urdf and all(abs(merged.get(k, 1e9) - v) < 1e-9 for k, v in expected.items()) \
                and set(urdf['u'].movable()) <= set(merged):
            break
    u = urdf.get('u')
    check('/robot_description recibido', u is not None)
    if u is None:
        sys.exit(1)
    movable = sorted(u.movable())
    diff = {k: (round(merged.get(k, float('nan')), 6), round(v, 6)) for k, v in expected.items()
            if not abs(merged.get(k, 1e9) - v) < 1e-9}
    check('/robot_v3/joint_states combina brazo + pinza + pan/tilt de la "Jetson"', not diff,
          f'distintos (recibido, esperado): {diff}' if diff else '')
    check('elevador y ruedas completados por defecto (0) para que exista toda la TF',
          set(movable) <= set(merged) and all(merged[j] == 0.0 for j in ('torso_lift_joint', 'left_wheel_joint')))
    for _ in range(40):
        publish()
        rclpy.spin_once(node, timeout_sec=0.05)
    q = {j: expected.get(j, 0.0) for j in movable}   # referencia: lo publicado por la "Jetson", no lo combinado
    for link in ('tcp_link', 'link7', 'link8', 'camera_link', 'mounting_cube_link'):
        try:
            t = buf.lookup_transform('base_footprint', link, Time())
        except Exception as exc:  # noqa: BLE001
            check(f'TF base_footprint -> {link}', False, str(exc))
            continue
        tr = t.transform.translation
        rot = quat_to_matrix(t.transform.rotation)
        ref = u.pose(link, q)
        err = np.linalg.norm(np.array([tr.x, tr.y, tr.z]) - ref[:3, 3])
        check(f'TF de {link} = cinemática del URDF con los estados de la "Jetson"',
              err < 1e-6 and np.allclose(rot, ref[:3, :3], atol=1e-6), f'{err * 1000:.4f} mm')
    names = set(u.links)
    driver_frames = [f for f in names if f.startswith('camera_') and f.endswith(('_optical_frame', 'color_frame', 'depth_frame'))]
    check('camera_frames:=false: el URDF no publica los frames que publica el driver de la D435', not driver_frames,
          str(driver_frames))
    node.destroy_node()
    rclpy.shutdown()
    print(f'\n{sum(results)}/{len(results)} chequeos de estados reales OK')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
