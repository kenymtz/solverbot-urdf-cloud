#!/usr/bin/env python3
"""Secuencia de prueba para la SIMULACIÓN (sim_mock.launch.py o gazebo.launch.py).

Envía trayectorias a los controladores simulados (lift_controller, arm_controller, head_controller,
gripper_controller). No habla con el robot real: la Jetson escucha /arm/joint_cmd, no estas acciones, y el script
se niega a correr en el dominio DDS del robot (ROS_DOMAIN_ID=10) salvo --allow-robot-domain.

    ros2 run robot_v3_description demo_motion.py
    ros2 run robot_v3_description demo_motion.py --only lift
"""
import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint

ROBOT_DOMAIN_ID = '10'
DEG = math.pi / 180.0
ARM = [f'joint{i}' for i in range(1, 7)]

# (controlador, joints, [(posiciones, segundos)])
SEQUENCES = {
    'lift': ('lift_controller', ['torso_lift_joint'], [([-0.15], 4.0), ([0.05], 5.0), ([0.0], 2.0)]),
    'head': ('head_controller', ['pan_joint', 'tilt_joint'],
             [([0.6, 0.0], 1.5), ([-0.6, 0.3], 2.0), ([0.0, 0.0], 1.5)]),
    'arm': ('arm_controller', ARM, [
        ([0.0, 1.2, -1.0, 0.0, 0.5, 0.0], 3.0),                                 # brazo extendido hacia afuera
        ([v * DEG for v in (86.28, 173.91, -164.0, 66.98, 0.69, -146.41)], 4.0),  # DEFAULT_HOME de la Jetson
        ([0.0] * 6, 3.0)]),
    # Una sola orden de apertura (joint7, por dedo); joint8 la sigue como mimic
    'gripper': ('gripper_controller', ['joint7'],
                [([0.0175], 1.0), ([0.035], 1.0), ([0.0], 1.5)]),
}


class Demo(Node):
    def __init__(self):
        super().__init__('robot_v3_demo_motion')
        self.limits = None
        self.js = {}
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/robot_description', self._on_description, qos)
        self.create_subscription(JointState, '/joint_states', lambda m: self.js.update(zip(m.name, m.position)), 10)

    def _on_description(self, msg):
        root = ET.fromstring(msg.data)
        self.limits = {j.get('name'): {k: float(j.find('limit').get(k)) for k in ('lower', 'upper', 'velocity')}
                       for j in root.findall('joint') if j.find('limit') is not None}

    def run(self, key):
        controller, joints, points = SEQUENCES[key]
        for positions, _ in points:
            for j, p in zip(joints, positions):
                lo, hi = self.limits[j]['lower'], self.limits[j]['upper']
                if not lo - 1e-9 <= p <= hi + 1e-9:
                    raise ValueError(f'{j}={p:.4f} fuera de los límites del URDF [{lo}, {hi}]')
        client = ActionClient(self, FollowJointTrajectory, f'/{controller}/follow_joint_trajectory')
        if not client.wait_for_server(timeout_sec=20.0):
            raise RuntimeError(f'{controller} no disponible (¿está corriendo sim_mock.launch.py o gazebo.launch.py?)')
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joints
        t = 0.0
        previous = [self.js.get(j, 0.0) for j in joints]
        for positions, dt in points:
            # Respetar la velocidad límite del URDF (pico de la spline cúbica ≈ 1.5 x la media). En Gazebo, si el
            # comando de velocidad supera el límite, DART lo descarta y el elevador cae al tope inferior.
            dt = max([dt] + [1.6 * abs(p - q) / self.limits[j]['velocity'] for j, p, q in zip(joints, positions, previous)])
            previous = positions
            t += dt
            goal.trajectory.points.append(JointTrajectoryPoint(
                positions=[float(p) for p in positions],
                time_from_start=Duration(seconds=t).to_msg()))
        self.get_logger().info(f'{controller}: {len(points)} puntos, {t:.1f} s')
        send = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send)
        handle = send.result()
        if not handle.accepted:
            raise RuntimeError(f'{controller} rechazó la trayectoria')
        res = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res, timeout_sec=t + 15.0)
        code = res.result().result.error_code if res.result() else None
        self.get_logger().info(f'{controller}: resultado {code} (0 = SUCCESSFUL)')
        return code == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=list(SEQUENCES), action='append')
    ap.add_argument('--allow-robot-domain', action='store_true')
    args, ros_args = ap.parse_known_args()
    if os.environ.get('ROS_DOMAIN_ID', '0') == ROBOT_DOMAIN_ID and not args.allow_robot_domain:
        sys.exit('ROS_DOMAIN_ID=10 es el dominio del robot real; use otro dominio para la simulación.')
    rclpy.init(args=ros_args)
    node = Demo()
    while rclpy.ok() and (node.limits is None or not node.js):
        rclpy.spin_once(node, timeout_sec=0.5)
    ok = all([node.run(k) for k in (args.only or ['lift', 'head', 'gripper', 'arm'])])
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
