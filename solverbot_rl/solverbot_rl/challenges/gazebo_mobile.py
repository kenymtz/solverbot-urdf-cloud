"""Backend físico de base móvil en Gazebo Harmonic (ROS 2 Jazzy, rclpy) para HRI, GPSR y Doing Laundry.

Mismo esquema de sincronización que el backend de Pick and Place (backends/gazebo.py, sin modificarlo): el mundo se
pausa y cada advance(dt) pide multi_step iteraciones por /world/<w>/control y espera reloj y mediciones nuevas.

Mediciones (las del robot real): odometría del diff_drive_controller, /scan, /joint_states, cámara de cabeza.
Verdad (solo métricas/expertos): /model/robot_v3/pose (PosePublisher, solo simulación).
Entidades del desafío (personas, hoja de la puerta de entrada, prendas): modelos del mundo movidos con
/world/<w>/set_pose por el simulador del desafío; su pose verdadera es la que se fija.
"""
from __future__ import annotations

import math
import os
import threading
import time

import numpy as np

from ..kinematics import UrdfChain
from .arena import AXLE, ArenaMap
from .backend import PAN_LIM, TILT_LIM, MobileBackend, compose, relative, wrap

try:
    import rclpy
    from geometry_msgs.msg import TwistStamped
    from nav_msgs.msg import Odometry
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from ros_gz_interfaces.srv import ControlWorld, SetEntityPose
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo, Image, JointState, LaserScan
    from std_msgs.msg import Float64MultiArray, String
    from tf2_msgs.msg import TFMessage
except ImportError as exc:  # pragma: no cover
    rclpy = None
    _IMPORT_ERROR = exc

# Brazo en DEFAULT_HOME (pose de reposo de la Jetson, plegado al costado de la caja) y elevador en 0: huella de 0.669 m
STOW = [math.radians(v) for v in (86.28, 173.91, -164.0, 66.98, 0.69, -146.41)]


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def _stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class GazeboMobileBackend(MobileBackend):
    physical = True

    def __init__(self, arena: ArenaMap, world: str = 'home_challenges', camera: bool = False, timeout: float = 30.0,
                 physics_step: float = 0.001, contact_margin: float = 0.005):
        if rclpy is None:
            raise RuntimeError(f'GazeboMobileBackend requiere ROS 2 Jazzy: {_IMPORT_ERROR}')
        super().__init__(arena, lidar_beams=271)
        # SOLVERBOT_GZ_TIMEOUT: espera máxima (s) por paso; con render por CPU (llvmpipe) y cámaras puede hacer falta más
        timeout = float(os.environ.get('SOLVERBOT_GZ_TIMEOUT', timeout))
        self.world, self.timeout, self.physics_step = world, timeout, physics_step
        self.contact_margin = contact_margin
        self.use_camera = camera
        self._lock = threading.Lock()
        self._clock = 0.0
        self._js, self._js_t = {}, -1.0
        self._odom_raw, self._odom_t = None, -1.0
        self._scan, self._scan_t = None, -1.0
        self._truth = None
        self._img = {}
        self._k = None
        self._urdf = None
        self._entities: dict[str, np.ndarray] = {}
        self._collided = False
        self._v = self._w = 0.0
        self._pan = self._tilt = 0.0
        self.stale = {'scan': 0, 'joint_states': 0, 'images': 0}

        if not rclpy.ok():
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self.node = n = Node('solverbot_challenge_env', parameter_overrides=[Parameter('use_sim_time', value=True)])
        be = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_cmd = n.create_publisher(TwistStamped, '/diff_drive_controller/cmd_vel', 10)
        self.pub_head = n.create_publisher(Float64MultiArray, '/head_position_controller/commands', 10)
        self.pub_arm = n.create_publisher(Float64MultiArray, '/arm_position_controller/commands', 10)
        self.pub_lift = n.create_publisher(Float64MultiArray, '/lift_position_controller/commands', 10)
        n.create_subscription(Clock, '/clock', self._on_clock, be)
        n.create_subscription(JointState, '/joint_states', self._on_js, 10)
        n.create_subscription(Odometry, '/diff_drive_controller/odom', self._on_odom, 10)
        n.create_subscription(LaserScan, '/scan', self._on_scan, be)
        n.create_subscription(TFMessage, '/model/robot_v3/pose', self._on_pose, be)
        n.create_subscription(String, '/robot_description', lambda m: setattr(self, '_urdf', m.data), latched)
        if camera:
            n.create_subscription(Image, '/camera/camera/color/image_raw', lambda m: self._on_img('rgb', m), be)
            n.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw',
                                  lambda m: self._on_img('depth', m), be)
            n.create_subscription(CameraInfo, '/camera/camera/color/camera_info',
                                  lambda m: setattr(self, '_k', np.array(m.k, float).reshape(3, 3)), be)
            # cámara de muñeca (virtual, solo simulación; ver gazebo_harmonic.xacro): Doing Laundry
            n.create_subscription(Image, '/wrist_camera/image_raw', lambda m: self._on_img('wrist', m), be)
            n.create_subscription(CameraInfo, '/wrist_camera/camera_info',
                                  lambda m: setattr(self, '_k_wrist', np.array(m.k, float).reshape(3, 3)), be)
        self.cli_ctrl = n.create_client(ControlWorld, f'/world/{world}/control')
        self.cli_pose = n.create_client(SetEntityPose, f'/world/{world}/set_pose')
        self.exec = MultiThreadedExecutor(num_threads=2)
        self.exec.add_node(n)
        self._spin = threading.Thread(target=self.exec.spin, daemon=True)
        self._spin.start()
        for cli in (self.cli_ctrl, self.cli_pose):
            if not cli.wait_for_service(timeout_sec=timeout):
                raise TimeoutError(f'Servicio {cli.srv_name} no disponible: ¿está corriendo la simulación del desafío?')
        run = ControlWorld.Request()
        run.world_control.pause = False
        self._call(self.cli_ctrl, run)
        self._wait(lambda: self._urdf is not None and self._js and self._odom_raw is not None and self._truth is not None
                   and self._scan is not None, 'robot_description, /joint_states, odometría, /scan y pose del robot')
        if camera:
            self._wait(lambda: 'rgb' in self._img and 'depth' in self._img and self._k is not None,
                       'imágenes de la cámara de cabeza')
        self.chain = UrdfChain(self._urdf)
        self._wait_controllers()
        pause = ControlWorld.Request()
        pause.world_control.pause = True
        self._call(self.cli_ctrl, pause)
        self._T_map_odom = np.zeros(3)

    # ----------------------------------------------------------------------------------------- ROS
    def _on_clock(self, m):
        self._clock = m.clock.sec + m.clock.nanosec * 1e-9

    def _on_js(self, m):
        with self._lock:
            self._js.update(zip(m.name, m.position))
            self._js_t = _stamp(m)

    def _on_odom(self, m):
        p = m.pose.pose
        self._odom_raw = np.array([p.position.x, p.position.y, _yaw(p.orientation)])
        self._odom_t = _stamp(m)

    def _on_scan(self, m):
        r = np.asarray(m.ranges, float)
        r[(r < m.range_min) | (r > m.range_max)] = np.inf
        if r.size != self.lidar_angles.size:
            self.lidar_angles = m.angle_min + m.angle_increment * np.arange(r.size)
        self._scan, self._scan_t = r, _stamp(m)

    def _on_pose(self, m):
        for t in m.transforms:
            if t.child_frame_id == 'robot_v3':
                q = t.transform.rotation
                self._truth = np.array([t.transform.translation.x, t.transform.translation.y, _yaw(q)])
                # inclinación (rad) respecto de la vertical: detecta vuelcos
                self._tilt_truth = math.acos(max(-1.0, min(1.0, 1 - 2 * (q.x * q.x + q.y * q.y))))

    def _on_img(self, key, m):
        if m.encoding == '32FC1':
            a = np.frombuffer(m.data, np.float32).reshape(m.height, m.width)
        else:
            a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[..., :3]
        self._img[key] = (_stamp(m), a)

    def _wait(self, cond, what, timeout=None):
        t0 = time.monotonic()
        while not cond():
            if time.monotonic() - t0 > (timeout or self.timeout):
                raise TimeoutError(f'Tiempo de espera agotado: {what}; reloj={self._clock:.3f}')
            time.sleep(0.002)

    def _call(self, cli, req):
        fut = cli.call_async(req)
        self._wait(fut.done, cli.srv_name)
        return fut.result()

    def _wait_controllers(self):
        from controller_manager_msgs.srv import ListControllers
        cli = self.node.create_client(ListControllers, '/controller_manager/list_controllers')
        need = {'joint_state_broadcaster', 'arm_position_controller', 'lift_position_controller',
                'head_position_controller', 'diff_drive_controller'}
        t0 = time.monotonic()
        while True:
            if cli.wait_for_service(timeout_sec=1.0):
                res = self._call(cli, ListControllers.Request())
                if need <= {c.name for c in res.controller if c.state == 'active'}:
                    return
            if time.monotonic() - t0 > 180:
                raise TimeoutError(f'Controladores no activos: {need}')
            time.sleep(0.5)

    def _pub(self, pub, data):
        pub.publish(Float64MultiArray(data=[float(v) for v in data]))

    def _set_pose(self, name, x, y, yaw, z):
        req = SetEntityPose.Request()
        req.entity.name = name
        req.entity.type = 2
        req.pose.position.x, req.pose.position.y, req.pose.position.z = float(x), float(y), float(z)
        req.pose.orientation.z, req.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return self._call(self.cli_pose, req).success

    # ----------------------------------------------------------------------------------------- interfaz
    def reset(self, rng, pose, entities):
        self.rng = rng
        self._v, self._w = 0.0, 0.0
        self._send_cmd()
        self._pub(self.pub_arm, STOW)
        self._pub(self.pub_lift, [0.0])
        self._pan, self._tilt = 0.0, 0.0
        self._pub(self.pub_head, [0.0, 0.0])
        for name, e in entities.items():
            self.set_entity(name, *e)
        # brazo plegado y base detenida ANTES de teletransportar (sin velocidades residuales)
        for _ in range(60):
            self.advance(0.1)
            q = np.array([self._js.get(f'joint{i}', 0.0) for i in range(1, 7)])
            if np.max(np.abs(q - STOW)) < 0.03 and abs(self._js.get('torso_lift_joint', 0.0)) < 0.01:
                break
        else:
            raise RuntimeError('reset: el brazo no llegó a la pose plegada')
        for _ in range(2):
            self._set_pose('robot_v3', pose[0], pose[1], pose[2], 0.01)
            self.advance(0.3)
        tr = self.pose_truth()
        if np.linalg.norm(tr[:2] - np.asarray(pose[:2])) > 0.03:
            raise RuntimeError(f'reset: la base no quedó en la pose pedida ({tr} vs {pose})')
        # odometría: el frame del mapa se ancla a la pose conocida del reset (como al iniciar en una posición marcada)
        o = self._odom_raw
        self._T_map_odom = np.array(pose, float), o.copy()
        self._collided = False

    def _send_cmd(self):
        m = TwistStamped()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.header.frame_id = 'base_footprint'
        m.twist.linear.x, m.twist.angular.z = float(self._v), float(self._w)
        self.pub_cmd.publish(m)

    def cmd_vel(self, v, w):
        self._v, self._w = float(v), float(w)

    def head(self, pan, tilt):
        self._pan, self._tilt = float(np.clip(pan, -PAN_LIM, PAN_LIM)), float(np.clip(tilt, -TILT_LIM, TILT_LIM))
        self._pub(self.pub_head, [self._pan, self._tilt])

    def set_entity(self, name, x, y, yaw=0.0, z=0.0):
        self._set_pose(name, x, y, yaw, z)
        self._entities[name] = np.array([x, y, yaw], float)

    def advance(self, dt):
        self._send_cmd()                         # mantiene viva la orden (cmd_vel_timeout = 0.5 s)
        req = ControlWorld.Request()
        req.world_control.pause = True
        req.world_control.multi_step = max(1, int(round(dt / self.physics_step)))
        t_goal = self._clock + dt - 0.5 * self.physics_step
        js_goal = self._js_t + 0.5 * dt
        time.sleep(0.003)
        clock0 = self._clock
        self._call(self.cli_ctrl, req)
        # con carga alta se observó una solicitud de multi_step sin efecto: si el reloj no se movió en 5 s, se reenvía
        t_start = time.monotonic()
        while True:
            try:
                self._wait(lambda: self._clock >= t_goal and self._js_t >= js_goal,
                           f'avance del tiempo simulado (dt={dt})', timeout=5.0)
                break
            except TimeoutError:
                if self._clock >= t_goal:
                    # reloj al día pero falta un /joint_states (mensaje perdido con carga alta): seguir con el último
                    # y contarlo como medición vencida (info['stale'])
                    self.stale['joint_states'] += 1
                    break
                if time.monotonic() - t_start > self.timeout:
                    raise TimeoutError(f'avance sin completar: reloj={self._clock:.3f} objetivo={t_goal:.3f} '
                                       f'joint_states={self._js_t:.3f} objetivo_js={js_goal:.3f}') from None
                if self._clock <= clock0 + 1e-9:
                    # el reloj no se movió en 5 s (solicitud perdida o cortada): pedir los pasos que faltan
                    self.stale['step_retries'] = self.stale.get('step_retries', 0) + 1
                    rest = ControlWorld.Request()
                    rest.world_control.pause = True
                    rest.world_control.multi_step = max(1, int(round((t_goal - self._clock) / self.physics_step)) + 1)
                    self._call(self.cli_ctrl, rest)
                clock0 = self._clock
        t0 = time.monotonic()
        while (self._scan_t < t_goal - 0.11 or self._odom_t < t_goal - 0.05) and time.monotonic() - t0 < 2.0:
            time.sleep(0.002)
        self.stale['scan'] += int(self._scan_t < t_goal - 0.11)
        if self.use_camera:
            while self._img.get('rgb', (-1,))[0] < t_goal - 0.21 and time.monotonic() - t0 < 2.0:
                time.sleep(0.002)
            self.stale['images'] += int(self._img.get('rgb', (-1,))[0] < t_goal - 0.21)
        tr = self.pose_truth()
        if self.arena.footprint_collides(tr[0], tr[1], tr[2], margin=-self.contact_margin,
                                         dynamic=self.dynamic_obstacles()):
            self._collided = True

    def odom(self):
        """Pose de base_footprint por odometría. diff_drive_controller integra el EJE de las ruedas (y lo reporta como
        base_footprint, que está 8.4 cm detrás): se ancla el eje a la pose del reset y se vuelve a base_footprint."""
        pose0, o0 = self._T_map_odom if isinstance(self._T_map_odom, tuple) else (np.zeros(3), np.zeros(3))
        axle0 = compose(pose0, np.array([AXLE[0], AXLE[1], 0.0]))
        axle = compose(axle0, relative(o0, self._odom_raw))
        return compose(axle, np.array([-AXLE[0], -AXLE[1], 0.0]))

    def scan(self):
        return self._scan.copy()

    def head_joints(self):
        return self._js.get('pan_joint', 0.0), self._js.get('tilt_joint', 0.0)

    def joint_positions(self):
        with self._lock:
            return dict(self._js)

    def head_rgbd(self):
        if not self.use_camera or 'rgb' not in self._img or 'depth' not in self._img:
            return None
        T = self.chain.link_pose('camera_color_optical_frame', self.joint_positions())
        return self._img['rgb'][1], self._img['depth'][1], self._k, T

    def wrist_rgb(self):
        """(rgb, K, T cámara de muñeca en base_footprint, sello de tiempo de la imagen) o None."""
        if not self.use_camera or 'wrist' not in self._img or getattr(self, '_k_wrist', None) is None:
            return None
        T = self.chain.link_pose('wrist_sim_camera_optical_frame', self.joint_positions())
        return self._img['wrist'][1], self._k_wrist, T, self._img['wrist'][0]

    def pose_truth(self):
        return self._truth.copy()

    def collided_truth(self):
        return self._collided

    def tipped_truth(self) -> bool:
        """Base inclinada > 15° (vuelco o a punto de volcar)."""
        return getattr(self, '_tilt_truth', 0.0) > math.radians(15)

    def dynamic_obstacles(self):
        from .backend import PERSON_RADIUS
        rad = getattr(self, 'entity_radius', {})
        return [(e[0], e[1], rad.get(k, PERSON_RADIUS)) for k, e in self._entities.items() if rad.get(k, PERSON_RADIUS) > 0]

    def close(self):
        try:
            self._v = self._w = 0.0
            self._send_cmd()
        finally:
            self.exec.remove_node(self.node)
            self.exec.shutdown(timeout_sec=2.0)
            self._spin.join(timeout=2.0)
            self.node.destroy_node()
            rclpy.try_shutdown()
