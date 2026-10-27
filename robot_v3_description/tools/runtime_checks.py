#!/usr/bin/env python3
"""Chequeos en EJECUCIÓN contra una simulación de robot_v3 (sim_mock.launch.py o gazebo.launch.py).

Verifica con el grafo ROS real (robot_state_publisher, ros2_control, tf2):
  * /robot_description publicado y /joint_states con las 13 articulaciones móviles (joint8 es mimic de joint7);
  * TF: todos los links alcanzables desde base_footprint y coincidentes con la cinemática del URDF;
  * elevador: al comandar torso_lift_joint, caja/brazo se trasladan exactamente en Z y la columna no;
  * brazo: tcp_link en TF = cinemática directa de los joint_states tras una trayectoria;
  * cabezal: tilt positivo hace mirar la cámara hacia abajo; frames ópticos de la D435;
  * (--sensors, solo Gazebo) imágenes RGB/profundidad, nube y /scan con sus frame_id.
No se conecta al robot real y se niega a correr en ROS_DOMAIN_ID=10 salvo --allow-robot-domain.

    ros2 run robot_v3_description runtime_checks.py [--tolerance 0.002] [--sensors]
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState, LaserScan, PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_v3 import DEFAULT_HOME_DEG, OPTICAL_R, Urdf  # noqa: E402

ROBOT_DOMAIN_ID = '10'


def quat_to_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Checks(Node):
    def __init__(self, tol, use_sim_time):
        super().__init__('robot_v3_runtime_checks',
                         parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', value=use_sim_time)])
        self.tol = tol
        self.urdf = None
        self.js = {}
        self.results = []
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/robot_description', lambda m: setattr(self, 'urdf', Urdf(m.data)), qos)
        self.create_subscription(JointState, '/joint_states', self._on_js, 20)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self, spin_thread=False)

    def _on_js(self, msg):
        self.js.update(dict(zip(msg.name, msg.position)))
        # ign_ros2_control 0.7 (Gazebo Fortress, Humble) publica el estado de una articulación mimic como
        # "<nombre>_mimic". Se usa ese valor (posición física simulada) para verificar la sincronización.
        for name, pos in zip(msg.name, msg.position):
            if name.endswith('_mimic'):
                self.js[name[:-len('_mimic')]] = pos

    def spin_for(self, seconds, until=None):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if until and until():
                return True
        return until() if until else True

    def check(self, name, ok, detail=''):
        self.results.append((name, bool(ok)))
        print(f"[{'OK' if ok else 'FALLA'}] {name}" + (f' — {detail}' if detail else ''), flush=True)
        return ok

    def tf(self, parent, child):
        try:
            t = self.buffer.lookup_transform(parent, child, Time())
        except Exception:  # noqa: BLE001
            return None
        m = np.eye(4)
        m[:3, :3] = quat_to_matrix(t.transform.rotation)
        tr = t.transform.translation
        m[:3, 3] = [tr.x, tr.y, tr.z]
        return m

    def safe_seconds(self, joints, positions, seconds):
        """Duración que respeta la velocidad límite del URDF (pico de la spline cúbica ≈ 1.5 x la media).

        En Gazebo, ign_ros2_control sigue la posición con un comando de velocidad; si este supera el límite de la
        articulación, DART lo descarta (el elevador cae al tope y queda trabado). En el robot real el límite también
        aplica, así que toda trayectoria debe respetarlo."""
        need = 0.0
        for j, p in zip(joints, positions):
            vmax = self.urdf.joints[j]['limit']['velocity']
            need = max(need, 1.6 * abs(p - self.js.get(j, p)) / vmax)
        return max(seconds, need)

    def move(self, controller, joints, positions, seconds):
        seconds = self.safe_seconds(joints, positions, seconds)
        client = ActionClient(self, FollowJointTrajectory, f'/{controller}/follow_joint_trajectory')
        if not client.wait_for_server(timeout_sec=30.0):
            return self.check(f'{controller} disponible', False)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joints
        goal.trajectory.points = [JointTrajectoryPoint(positions=[float(p) for p in positions],
                                                       time_from_start=Duration(seconds=seconds).to_msg())]
        fut = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        if not fut.result() or not fut.result().accepted:
            return self.check(f'{controller} acepta la trayectoria', False)
        res = fut.result().get_result_async()
        rclpy.spin_until_future_complete(self, res, timeout_sec=seconds + 30.0)
        code = res.result().result.error_code if res.result() else 'timeout'
        reached = self.spin_for(5.0, lambda: all(abs(self.js.get(j, 1e9) - p) < max(self.tol, 1e-3)
                                                for j, p in zip(joints, positions)))
        self.spin_for(1.0)   # dejar que llegue la TF correspondiente
        return self.check(f'{controller}: trayectoria ejecutada y estado alcanzado', code == 0 and reached,
                          f'error_code={code}, estado={[round(self.js.get(j, float("nan")), 4) for j in joints]}')

    def fk_matches_tf(self, label):
        worst, worst_link, missing = 0.0, '', []
        for link in self.urdf.links:
            t = self.tf('base_footprint', link)
            if t is None:
                missing.append(link)
                continue
            err = np.linalg.norm(t[:3, 3] - self.urdf.pose(link, self.js)[:3, 3])
            if err > worst:
                worst, worst_link = err, link
        self.check(f'{label}: TF de los {len(self.urdf.links)} links = cinemática del URDF con los /joint_states',
                   not missing and worst < self.tol, f'máx {worst * 1000:.3f} mm ({worst_link})' +
                   (f'; sin TF: {missing}' if missing else ''))

    def run(self, sensors):
        self.check('/robot_description recibido', self.spin_for(30.0, lambda: self.urdf is not None))
        if self.urdf is None:
            return
        movable = sorted(n for n, j in self.urdf.joints.items() if j['type'] != 'fixed')   # incluye joint8 (mimic)
        self.check('/joint_states con las 13 articulaciones móviles (joint8 puede llegar como joint8_mimic en Gazebo)',
                   self.spin_for(30.0, lambda: set(movable) <= set(self.js)), f'{sorted(self.js)}')
        self.spin_for(30.0, lambda: all(self.tf('base_footprint', link) is not None for link in self.urdf.links))
        self.fk_matches_tf('estado inicial')

        # Elevador
        cube0, cam0 = self.tf('base_footprint', 'mounting_cube_link'), self.tf('base_footprint', 'camera_link')
        col0 = self.tf('base_footprint', 'torso_column')
        head_on_lift = 'torso_lift_link' in self.urdf.ancestors('pan_tilt_base_link')
        for target in (0.05, -0.15, 0.0):
            self.move('lift_controller', ['torso_lift_joint'], [target], 3.0)
            cube, cam = self.tf('base_footprint', 'mounting_cube_link'), self.tf('base_footprint', 'camera_link')
            col = self.tf('base_footprint', 'torso_column')
            dz = cube[:3, 3] - cube0[:3, 3]
            dcam = cam[:3, 3] - cam0[:3, 3]
            exp_cam = [0, 0, target] if head_on_lift else [0, 0, 0]
            self.check(f'elevador en {target:+.2f} m: caja se mueve {dz[2]:+.4f} m solo en Z; columna quieta; '
                       f'cámara {"sube con el carro" if head_on_lift else "quieta"}',
                       np.allclose(dz, [0, 0, target], atol=self.tol) and np.allclose(col, col0, atol=1e-6) and
                       np.allclose(dcam, exp_cam, atol=self.tol), f'Δcámara={np.round(dcam, 4)}')
            self.fk_matches_tf(f'elevador {target:+.2f} m')

        # Brazo y pinza
        home = [math.radians(v) for v in DEFAULT_HOME_DEG]
        self.move('arm_controller', [f'joint{i}' for i in range(1, 7)], home, 4.0)
        tcp = self.tf('base_footprint', 'tcp_link')
        err = np.linalg.norm(tcp[:3, 3] - self.urdf.pose('tcp_link', self.js)[:3, 3])
        self.check('brazo en DEFAULT_HOME: tcp_link en TF = cinemática directa', err < self.tol, f'{err * 1000:.3f} mm')
        # Pinza: una sola orden (joint7); joint8 debe seguirla como mimic (= -joint7). Mitad, abierta y cerrada.
        for a, label in ((0.0175, 'a mitad'), (0.035, 'abierta'), (0.0, 'cerrada')):
            self.move('gripper_controller', ['joint7'], [a], 1.5)
            j8 = self.js.get('joint8', float('nan'))
            d = np.linalg.norm(self.tf('link7', 'link8')[:3, 3])
            self.check(f'pinza {label} (solo joint7 = {a * 1000:.1f} mm): joint8 = -joint7 y dedos separados {2 * a * 1000:.0f} mm',
                       abs(j8 + a) < max(self.tol, 1e-3) and abs(d - 2 * a) < 2 * max(self.tol, 1e-3),
                       f'joint8 = {j8 * 1000:.2f} mm, separación {d * 1000:.2f} mm')
            if a == 0.035:
                self.fk_matches_tf('brazo en DEFAULT_HOME, pinza abierta')
        self.move('arm_controller', [f'joint{i}' for i in range(1, 7)], [0.0] * 6, 3.0)

        # Cabezal y frames ópticos
        self.move('head_controller', ['pan_joint', 'tilt_joint'], [0.0, 0.3], 1.5)
        opt = self.tf('base_footprint', 'camera_color_optical_frame')
        if opt is None:
            self.check('camera_color_optical_frame en TF (camera_frames:=true)', False)
        else:
            look = opt[:3, 2]
            self.check('tilt +0.3 rad: el eje óptico (z) apunta hacia adelante y abajo',
                       look[0] > 0.9 and abs(look[2] + math.sin(0.3)) < 0.01, f'z óptico = {np.round(look, 3)}')
            rel = self.tf('camera_color_frame', 'camera_color_optical_frame')
            self.check('camera_color_frame -> camera_color_optical_frame con convención REP-103',
                       np.allclose(rel[:3, :3], OPTICAL_R, atol=1e-6))
        self.move('head_controller', ['pan_joint', 'tilt_joint'], [0.0, 0.0], 1.5)

        if sensors:
            self.sensor_checks()

    def sensor_checks(self):
        got = {}

        def keep(key):
            return lambda m: got.setdefault(key, m)
        self.create_subscription(Image, '/camera/camera/color/image_raw', keep('color'), qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', keep('depth'),
                                 qos_profile_sensor_data)
        self.create_subscription(PointCloud2, '/camera/camera/depth/color/points', keep('points'), qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', keep('scan'), qos_profile_sensor_data)
        self.spin_for(60.0, lambda: len(got) == 4)
        for key, frame in (('color', 'camera_color_optical_frame'), ('depth', 'camera_color_optical_frame'),
                           ('points', 'camera_color_optical_frame'), ('scan', 'laser')):
            m = got.get(key)
            self.check(f'sensor simulado {key}: recibido con frame_id={frame}', m is not None and m.header.frame_id == frame,
                       '' if m is None else f'frame_id={m.header.frame_id}')
        if 'depth' in got:
            d = got['depth']
            arr = np.frombuffer(bytes(d.data), dtype=np.float32 if d.encoding == '32FC1' else np.uint16)
            arr = arr.reshape(d.height, d.width)
            valid = arr[np.isfinite(arr) & (arr > 0)]
            frac = valid.size / arr.size
            # Con el cabezal en cero la mitad superior mira al vacío (sin paredes): se exige que el piso y la mesa
            # (mitad inferior) den profundidades finitas y plausibles
            self.check('profundidad simulada: >20 % de píxeles con distancia finita, mediana entre 0.3 y 10 m',
                       frac > 0.2 and 0.3 < float(np.median(valid)) < 10.0,
                       f'encoding={d.encoding}, {frac * 100:.0f} % válidos, mediana {np.median(valid):.2f} m'
                       if valid.size else d.encoding)
        if 'scan' in got:
            r = np.asarray(got['scan'].ranges)
            self.check('LiDAR simulado: 1081 haces y alguna lectura dentro de rango', r.size == 1081 and np.isfinite(r).any(),
                       f'{r.size} haces, mínimo {np.nanmin(np.where(np.isfinite(r), r, np.nan)):.2f} m')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tolerance', type=float, default=0.002, help='tolerancia de posición [m] (Gazebo: ~0.005)')
    ap.add_argument('--sensors', action='store_true', help='verificar sensores simulados (solo gazebo.launch.py)')
    ap.add_argument('--use-sim-time', action='store_true', help='usar /clock (Gazebo)')
    ap.add_argument('--allow-robot-domain', action='store_true')
    args, ros_args = ap.parse_known_args()
    if os.environ.get('ROS_DOMAIN_ID', '0') == ROBOT_DOMAIN_ID and not args.allow_robot_domain:
        sys.exit('ROS_DOMAIN_ID=10 es el dominio del robot real; use otro dominio para la simulación.')
    rclpy.init(args=ros_args)
    node = Checks(args.tolerance, args.use_sim_time)
    try:
        node.run(args.sensors)
    finally:
        ok = sum(r[1] for r in node.results)
        print(f'\n{ok}/{len(node.results)} chequeos en ejecución OK', flush=True)
        failed = ok != len(node.results) or not node.results
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
