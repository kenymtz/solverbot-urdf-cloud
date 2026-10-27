"""Backend Gazebo Harmonic (ROS 2 Jazzy) por rclpy: comandos a ros2_control, tiempo simulado controlado
paso a paso y sensores reales de la simulación.

Requisitos: `ros2 launch solverbot_sim sim.launch.py world:=pick_place_training headless:=true cameras:=false
camera_profile:=train` ya en marcha en el MISMO ROS_DOMAIN_ID / GZ_PARTITION (scripts/run_sim.sh lo hace).

Sincronización: al conectarse, el backend espera a que los controladores estén activos y PAUSA el mundo; cada `advance(dt)` pide `multi_step` iteraciones de física por
/world/<w>/control y espera a que /clock, /joint_states y las imágenes alcancen el nuevo instante.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from ..kinematics import UrdfChain
from ..perception import locate_on_plane
from .base import Backend

try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from ros_gz_interfaces.srv import ControlWorld, SetEntityPose
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo, Image, JointState
    from std_msgs.msg import Float64MultiArray, String
    from tf2_msgs.msg import TFMessage
except ImportError as exc:  # pragma: no cover - sin ROS (p. ej. CI o el backend cinemático)
    rclpy = None
    _IMPORT_ERROR = exc


def _stamp(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def _image_to_np(msg) -> np.ndarray:
    if msg.encoding in ('32FC1',):
        return np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
    return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)[..., :3]


class GazeboBackend(Backend):
    physical = True

    def __init__(self, cfg: dict, world: str = 'pick_place_training', images: tuple[str, ...] = (),
                 perception: bool = True, timeout: float = 30.0):
        if rclpy is None:
            raise RuntimeError(f'GazeboBackend requiere ROS 2 Jazzy (rclpy, ros_gz_interfaces): {_IMPORT_ERROR}')
        self.cfg = cfg
        self.world = world
        self.obj_name = cfg['world']['object']['name']
        self.obj_rgb = cfg['world']['object']['rgb']
        self.obj_half_h = cfg['world']['object']['size'][2] / 2
        self.table_h = cfg['task']['table_height']
        self.physics_step = cfg['world']['physics_step']
        self.arm_joints = cfg['robot']['arm_joints']
        self.want_images = set(images)
        self.use_perception = perception
        self.timeout = timeout
        self.max_image_age = 0.2                # s de tiempo simulado
        self.stale_images = 0
        self._lock = threading.Lock()
        self._clock = 0.0
        self._js: dict[str, float] = {}
        self._js_t = -1.0
        self._obj_pose: np.ndarray | None = None
        self._base_pose: np.ndarray | None = None      # 4x4 base del robot en el mundo (verdad)
        self._img: dict[str, tuple[float, np.ndarray]] = {}
        self._k: np.ndarray | None = None
        self._urdf: str | None = None

        if not rclpy.ok():
            rclpy.init()
        self.node = Node('solverbot_rl_env', parameter_overrides=[Parameter('use_sim_time', value=True)])
        n = self.node
        be = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_arm = n.create_publisher(Float64MultiArray, '/arm_position_controller/commands', 10)
        self.pub_grip = n.create_publisher(Float64MultiArray, '/gripper_position_controller/commands', 10)
        self.pub_lift = n.create_publisher(Float64MultiArray, '/lift_position_controller/commands', 10)
        self.pub_head = n.create_publisher(Float64MultiArray, '/head_position_controller/commands', 10)
        n.create_subscription(Clock, '/clock', self._on_clock, be)
        n.create_subscription(JointState, '/joint_states', self._on_js, 10)
        n.create_subscription(TFMessage, f'/model/{self.obj_name}/pose', self._on_poses, be)
        n.create_subscription(TFMessage, '/model/robot_v3/pose', self._on_poses, be)
        n.create_subscription(String, '/robot_description', lambda m: setattr(self, '_urdf', m.data), latched)
        topics = {'head': '/camera/camera/color/image_raw', 'depth': '/camera/camera/aligned_depth_to_color/image_raw',
                  'wrist': '/wrist_camera/image_raw'}
        # Percepción del objeto: cámara de muñeca (la D435 de cabeza queda tapada por la placa superior de la columna
        # al mirar la mesa de costado; ver docs/HALLAZGOS.md)
        need = set(self.want_images) | ({'wrist'} if perception else set())
        for key in need:
            n.create_subscription(Image, topics[key], lambda m, k=key: self._on_img(k, m), be)
        if perception:
            n.create_subscription(CameraInfo, '/wrist_camera/camera_info', self._on_info, be)
        self.cli_ctrl = n.create_client(ControlWorld, f'/world/{world}/control')
        self.cli_pose = n.create_client(SetEntityPose, f'/world/{world}/set_pose')
        self.exec = MultiThreadedExecutor(num_threads=2)
        self.exec.add_node(n)
        self._spin = threading.Thread(target=self.exec.spin, daemon=True)
        self._spin.start()

        for cli in (self.cli_ctrl, self.cli_pose):
            if not cli.wait_for_service(timeout_sec=timeout):
                raise TimeoutError(f'Servicio {cli.srv_name} no disponible: ¿está corriendo sim.launch.py?')
        run = ControlWorld.Request()
        run.world_control.pause = False         # por si quedó en pausa de una conexión anterior
        self._call(self.cli_ctrl, run)
        self._wait(lambda: self._urdf is not None and self._js, 'robot_description y /joint_states')
        self.chain = UrdfChain(self._urdf)
        self._last_est = np.zeros(3)
        # El launch arranca el mundo corriendo (los spawners necesitan ciclos de control para activar los
        # controladores); a partir de aquí el tiempo solo avanza con advance().
        self._wait_controllers()
        req = ControlWorld.Request()
        req.world_control.pause = True
        self._call(self.cli_ctrl, req)

    def _wait_controllers(self):
        from controller_manager_msgs.srv import ListControllers
        cli = self.node.create_client(ListControllers, '/controller_manager/list_controllers')
        need = {'joint_state_broadcaster', 'arm_position_controller', 'gripper_position_controller',
                'lift_position_controller', 'head_position_controller'}
        t0 = time.monotonic()
        while True:
            if cli.wait_for_service(timeout_sec=1.0):
                res = self._call(cli, ListControllers.Request())
                if need <= {c.name for c in res.controller if c.state == 'active'}:
                    return
            if time.monotonic() - t0 > 180:
                raise TimeoutError(f'Controladores no activos: {need}')
            time.sleep(0.5)

    # ----------------------------------------------------------------------------------- callbacks
    def _on_clock(self, m):
        self._clock = m.clock.sec + m.clock.nanosec * 1e-9

    def _on_js(self, m):
        with self._lock:
            self._js.update(zip(m.name, m.position))
            self._js_t = _stamp(m)

    def _on_poses(self, m):
        for t in m.transforms:
            p, q = t.transform.translation, t.transform.rotation
            if t.child_frame_id == self.obj_name:
                self._obj_pose = np.array([p.x, p.y, p.z])
            elif t.child_frame_id == 'robot_v3':
                x, y, z, w = q.x, q.y, q.z, q.w
                m4 = np.eye(4)
                m4[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                              [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                              [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
                m4[:3, 3] = [p.x, p.y, p.z]
                self._base_pose = m4

    def _on_img(self, key, m):
        self._img[key] = (_stamp(m), _image_to_np(m))

    def _on_info(self, m):
        self._k = np.array(m.k, float).reshape(3, 3)

    def _wait(self, cond, what, timeout=None):
        t0 = time.monotonic()
        while not cond():
            if time.monotonic() - t0 > (timeout or self.timeout):
                raise TimeoutError(f'Tiempo de espera agotado: {what}')
            time.sleep(0.002)

    def _call(self, cli, req):
        fut = cli.call_async(req)
        self._wait(fut.done, cli.srv_name)
        return fut.result()

    # ----------------------------------------------------------------------------------- interfaz
    def _pub(self, pub, values):
        pub.publish(Float64MultiArray(data=[float(v) for v in values]))

    def command(self, arm_q, gripper):
        self._pub(self.pub_arm, arm_q)
        self._pub(self.pub_grip, [gripper])

    def advance(self, dt):
        req = ControlWorld.Request()
        req.world_control.pause = True
        req.world_control.multi_step = max(1, int(round(dt / self.physics_step)))
        t_goal = self._clock + dt - 0.5 * self.physics_step
        time.sleep(0.003)                       # deja salir los comandos publicados antes de avanzar
        self._call(self.cli_ctrl, req)
        # joint_state_broadcaster publica en ciclos de control (100 Hz): puede ir hasta ~3 ciclos detrás de /clock
        self._wait(lambda: self._clock >= t_goal and self._js_t >= t_goal - 0.035, 'avance del tiempo simulado')
        # Imágenes: con render por CPU, Gazebo puede saltear cuadros. Se espera un cuadro reciente un tiempo acotado
        # y, si no llega, se usa el último y se cuenta como imagen vieja (info['stale_images'] en el entorno).
        self.stale_images = 0
        for key in self._img:
            t0 = time.monotonic()
            while self._img[key][0] < t_goal - self.max_image_age and time.monotonic() - t0 < 2.0:
                time.sleep(0.002)
            self.stale_images += int(self._img[key][0] < t_goal - self.max_image_age)

    def _base(self):
        self._wait(lambda: self._base_pose is not None, 'pose de robot_v3 en /model/robot_v3/pose')
        return self._base_pose.copy()

    def set_pose_in_base(self, name, xyz, yaw=0.0):
        """Ubica un modelo con coordenadas relativas a la base del robot (la base puede haberse desplazado)."""
        b = self._base()
        w = b @ np.append(np.asarray(xyz, float), 1.0)
        return self.set_pose(name, [w[0], w[1], xyz[2]], yaw + float(np.arctan2(b[1, 0], b[0, 0])))

    def set_pose(self, name, xyz, yaw=0.0):
        req = SetEntityPose.Request()
        req.entity.name = name
        req.entity.type = 2                     # MODEL
        req.pose.position.x, req.pose.position.y, req.pose.position.z = map(float, xyz)
        req.pose.orientation.z, req.pose.orientation.w = float(np.sin(yaw / 2)), float(np.cos(yaw / 2))
        return self._call(self.cli_pose, req).success

    def reset(self, rng, obj_xy, obj_yaw, arm_q, gripper, lift, head):
        # Se lleva el robot a la pose inicial con sus controladores (sin teletransportar articulaciones) y
        # luego se reubica el objeto: el estado del episodio anterior no se arrastra.
        self.set_pose_in_base(self.obj_name, [obj_xy[0], -0.8, self.obj_half_h + 0.002], obj_yaw)   # al piso, lejos
        self._pub(self.pub_lift, [lift])
        self._pub(self.pub_head, [head['pan_joint'], head['tilt_joint']])
        self.command(arm_q, gripper)
        self.reset_error = np.inf
        for _ in range(150):                    # hasta 15 s simulados
            self.advance(0.1)
            q = np.array([self._js.get(j, 0.0) for j in self.arm_joints])
            self.reset_error = float(max(np.max(np.abs(q - arm_q)), abs(self._js.get('torso_lift_joint', 0) - lift)))
            if self.reset_error < 0.02:
                break
        if self.reset_error >= 0.02:
            raise RuntimeError(f'reset: el robot no alcanzó la pose inicial (error {self.reset_error:.3f} rad/m)')
        self.set_pose_in_base(self.obj_name, [obj_xy[0], obj_xy[1], self.table_h + self.obj_half_h + 0.003], obj_yaw)
        self.advance(0.3)                       # el objeto se asienta sobre la mesa
        self._last_est = np.zeros(3)

    def joint_positions(self):
        with self._lock:
            return dict(self._js)

    def object_pose_truth(self):
        """Posición real del objeto en el frame de la BASE del robot (como la FK del TCP)."""
        self._wait(lambda: self._obj_pose is not None, f'pose de {self.obj_name} en /model/{self.obj_name}/pose')
        b = self._base()
        return (np.linalg.inv(b) @ np.append(self._obj_pose, 1.0))[:3]

    def object_estimate(self):
        """Estimación con la cámara de muñeca sobre el plano de la mesa. Solo se actualiza con la pinza abierta
        (con el objeto agarrado la hipótesis 'apoyado en la mesa' no vale); si no, devuelve la última."""
        js = self.joint_positions()
        if 'wrist' not in self._img or js.get('joint7', 0.0) < 0.025:
            return self._last_est.copy(), False
        cam = self.chain.link_pose('wrist_sim_camera_optical_frame', js)
        p = locate_on_plane(self._img['wrist'][1], self._k, cam, self.obj_rgb, self.table_h + self.obj_half_h)
        if p is None:
            return self._last_est.copy(), False
        self._last_est = p
        return p.copy(), True

    def images(self):
        return {k: v[1] for k, v in self._img.items() if k in self.want_images}

    def close(self):
        self.exec.shutdown()
        self.node.destroy_node()
