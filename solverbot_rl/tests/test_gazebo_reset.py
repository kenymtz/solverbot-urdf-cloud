"""GazeboBackend.reset() contra un mundo falso, sin ROS ni Gazebo. Solo se reemplazan la física y el transporte: las poses
y los estados articulares entran por los callbacks reales y las órdenes salen por el set_pose real."""
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from solverbot_rl.backends import gazebo
from solverbot_rl.backends.gazebo import GazeboBackend

JOINTS = [f'joint{i}' for i in range(1, 7)]
HOME = np.zeros(6)
HEAD = {'pan_joint': 0.0, 'tilt_joint': 0.0}
REST_Z = 0.00048
ROLL, PITCH = np.radians(-0.175), np.radians(0.233)


def quat(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (f(a / 2) for a in (roll, pitch, yaw) for f in (np.cos, np.sin))
    return np.array([sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy])


def yaw_of(q):
    x, y, z, w = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def tilt_of(q):
    x, y, z, w = q
    return (float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))), float(np.arcsin(2 * (w * y - z * x))))


def stamp(t):
    sec = int(t)
    return NS(sec=sec, nanosec=int(round((t - sec) * 1e9)))


@pytest.fixture(autouse=True)
def set_entity_pose(monkeypatch):
    """Con ROS se usa el mensaje real; sin ROS, uno con los mismos campos."""
    if not hasattr(gazebo, 'SetEntityPose'):
        def request():
            return NS(entity=NS(name='', type=0),
                      pose=NS(position=NS(x=0.0, y=0.0, z=0.0), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0)))
        monkeypatch.setattr(gazebo, 'SetEntityPose', NS(Request=request), raising=False)


class FakeWorld(GazeboBackend):
    """Reproduce lo medido: el regreso del brazo desliza la base, un teletransporte con roll=pitch=0 aterriza girado y las
    poses solo se publican cuando avanza la simulación."""

    pub_lift = pub_head = None

    def __init__(self, xy=(0.0, 0.0), yaw=0.0, tilt=(ROLL, PITCH), rest_tilt=(ROLL, PITCH), slide=0.01, obeys=lambda n: True,
                 offset=(0.0, 0.0)):
        self.obj_name, self.obj_half_h, self.table_h, self.arm_joints = 'obj', 0.02, 0.4, JOINTS
        self.timeout, self.cli_pose, self._lock = 0.2, 'cli_pose', threading.Lock()
        self._clock, self._js, self._js_vel, self._js_t = 10.0, {}, {}, -1.0
        self._base_pose, self._base_q, self._base_t = None, np.array([0.0, 0.0, 0.0, 1.0]), -1.0
        self.q, self.target, self.speed = np.ones(6), np.ones(6), 0.0
        self.xyz, self.yaw, self.tilt, self.rest_tilt = np.array([*xy, REST_Z]), yaw, tilt, rest_tilt
        self.slide, self.obeys, self.offset = slide, obeys, offset
        self.landing, self.log, self.advances, self.obj_table_base = None, [], 0, None
        self.lift_speed, self.joint_speeds = 0.0, True
        self.emit_joints()
        self.emit_pose()

    # ------------------------------------------------------------------ lo que publica el simulador
    def emit_pose(self, age=0.0):
        q = quat(*self.tilt, self.yaw)
        self._on_poses(NS(transforms=[NS(
            child_frame_id='robot_v3', header=NS(stamp=stamp(self._clock - age)),
            transform=NS(translation=NS(x=self.xyz[0], y=self.xyz[1], z=self.xyz[2]),
                         rotation=NS(x=q[0], y=q[1], z=q[2], w=q[3])))]))

    def emit_joints(self):
        names = [*JOINTS, 'torso_lift_joint']
        velocity = [*[self.speed] * 6, self.lift_speed] if self.joint_speeds else []
        self._on_js(NS(name=names, position=[*self.q, 0.0], velocity=velocity, header=NS(stamp=stamp(self._clock))))

    def publish_pose(self):
        self.emit_pose()

    # ------------------------------------------------------------------ física
    def blocked(self):
        return False

    def advance(self, dt):
        self.advances += 1
        self._clock += dt
        step = np.zeros(6) if self.blocked() else (self.target - self.q) * 0.5
        self.q = self.q + step
        self.speed = float(np.max(np.abs(step))) / dt
        if self.speed > 1.0:                    # el brazo rápido desliza la base
            self.xyz[0] -= self.slide
            self.yaw += np.radians(200 * self.slide)
        elif self.speed > 0.01:                 # el brazo que todavía frena la desliza un poco
            self.xyz[0] -= self.slide / 20
        if self.landing is not None:
            self.xyz, self.yaw, self.tilt = self.landing
            self.landing = None
        self.emit_joints()
        self.publish_pose()

    # ------------------------------------------------------------------ transporte
    def _wait(self, cond, what, timeout=None):
        return super()._wait(cond, what, timeout=0.01)

    def _pub(self, pub, values):
        pass

    def command(self, arm_q, gripper):
        self.target = np.asarray(arm_q, float)

    def _call(self, cli, req):
        assert cli == 'cli_pose'
        p, o = req.pose.position, req.pose.orientation
        q = np.array([o.x, o.y, o.z, o.w])
        call = NS(name=req.entity.name, arm_err=float(np.max(np.abs(self.q - HOME))), speed=self.speed,
                  xyz=np.array([p.x, p.y, p.z]), q=q)
        self.log.append(call)
        self.on_set_pose(call)
        return NS(success=True)

    def on_set_pose(self, c):
        if c.name == self.obj_name and c.xyz[2] > self.table_h:
            self.obj_table_base = self._base_pose.copy()
        if c.name != 'robot_v3' or not self.obeys(len(self.robot_calls())):
            return
        if np.allclose(tilt_of(c.q), 0.0, atol=1e-9):   # roll=pitch=0: se reacomoda a su inclinación de reposo y gira
            self.landing = (np.array([c.xyz[0] - 0.002, c.xyz[1] + 0.0016, REST_Z]), np.radians(0.7), self.rest_tilt)
        else:
            self.landing = (c.xyz + [self.offset[0], 0.0, 0.0], yaw_of(c.q) + self.offset[1], tilt_of(c.q))

    def robot_calls(self):
        return [c for c in self.log if c.name == 'robot_v3']

    def run_reset(self):
        self.reset(None, [0.3, 0.0], 0.0, HOME, 0.03, 0.0, HEAD)


def test_base_se_reubica_despues_de_que_el_brazo_converge():
    be = FakeWorld()
    be.run_reset()
    assert be.robot_calls(), 'la base no se reubicó'
    assert all(c.arm_err < 0.02 for c in be.robot_calls())
    assert np.linalg.norm(be._base_pose[:2, 3]) <= 0.005
    assert abs(be.yaw) <= np.radians(0.5)


def test_reubicacion_conserva_altura_e_inclinacion_de_reposo():
    be = FakeWorld(xy=(-0.018, -0.003), yaw=np.radians(3.8))
    be.run_reset()
    c = be.robot_calls()[-1]
    assert c.xyz[:2] == pytest.approx([0.0, 0.0])
    assert c.xyz[2] == pytest.approx(REST_Z)
    assert yaw_of(c.q) == pytest.approx(0.0, abs=1e-9)
    assert tilt_of(c.q) == pytest.approx((ROLL, PITCH), abs=1e-6)


def test_exito_del_servicio_no_es_pose_verificada():
    be = FakeWorld(xy=(-0.008, 0.0), obeys=lambda n: False)
    with pytest.raises(RuntimeError, match='base'):
        be.run_reset()
    assert len(be.robot_calls()) == 3           # intentos acotados


def test_segundo_intento_corrige_un_pedido_perdido():
    be = FakeWorld(xy=(-0.008, 0.0), obeys=lambda n: n >= 2)
    be.run_reset()
    assert len(be.robot_calls()) == 2
    assert np.linalg.norm(be._base_pose[:2, 3]) <= 0.005


def test_base_volcada_cae_plana_y_luego_corrige_el_yaw():
    be = FakeWorld(tilt=(np.radians(20.0), 0.0))
    be.run_reset()
    calls = be.robot_calls()
    assert np.allclose(tilt_of(calls[0].q), 0.0)    # plano: no se conserva una inclinación anómala
    assert tilt_of(calls[-1].q) == pytest.approx((ROLL, PITCH), abs=1e-6)
    assert abs(be.yaw) <= np.radians(0.5) and abs(be.tilt[0]) <= np.radians(0.5)


def test_base_volcada_se_endereza_antes_de_mover_el_brazo():
    class BrazoTrabado(FakeWorld):
        """Volcada, el brazo no puede volver: solo converge cuando la base ya está derecha."""

        def blocked(self):
            return abs(self.tilt[0]) > np.radians(0.5)

    be = BrazoTrabado(tilt=(np.radians(20.0), 0.0))
    be.run_reset()
    assert be.robot_calls()[0].arm_err > 0.02   # se enderezó con el brazo todavía lejos de la pose inicial
    assert abs(be.tilt[0]) <= np.radians(0.5)


def test_base_derivada_se_suelta_antes_de_mover_el_brazo():
    class TrabadoHastaLaCaida(FakeWorld):
        """Medido con joint1 junto a su límite: no vuelve hasta que la base cae desde 1 cm; corregir el yaw no lo destraba."""

        def blocked(self):
            return not any(c.xyz[2] >= 0.01 for c in self.robot_calls())

    be = TrabadoHastaLaCaida(yaw=np.radians(2.3))
    be.run_reset()
    calls = be.robot_calls()
    assert calls[0].xyz[2] >= 0.01 and calls[0].arm_err > 0.02
    assert tilt_of(calls[-1].q) == pytest.approx((ROLL, PITCH), abs=1e-6)
    assert np.linalg.norm(be._base_pose[:2, 3]) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize('xy, yaw, dropped', [(0.009, 0.0, False), (0.011, 0.0, True), (0.0, 0.9, False), (0.0, 1.1, True)])
def test_umbral_de_la_suelta_previa(xy, yaw, dropped):
    be = FakeWorld(xy=(xy, 0.0), yaw=np.radians(yaw), slide=0.0)
    be.run_reset()
    assert (be.robot_calls()[0].xyz[2] >= 0.01) == dropped


def test_objeto_se_ubica_respecto_de_la_base_ya_reubicada():
    be = FakeWorld(xy=(-0.02, 0.0), yaw=np.radians(3.0))
    be.run_reset()
    assert np.linalg.norm(be.obj_table_base[:2, 3]) <= 0.005


def test_error_si_la_base_se_mueve_despues_de_verificarla():
    class Empujada(FakeWorld):
        def on_set_pose(self, c):
            super().on_set_pose(c)
            if c.name == self.obj_name and c.xyz[2] > self.table_h:
                self.xyz[0] -= 0.02             # algo la empuja al apoyar el objeto

    with pytest.raises(RuntimeError, match='base'):
        Empujada().run_reset()


def test_error_si_las_articulaciones_no_convergen_sin_tocar_la_base():
    class Trabado(FakeWorld):
        def blocked(self):
            return True

    be = Trabado()
    with pytest.raises(RuntimeError, match='pose inicial'):
        be.run_reset()
    assert not be.robot_calls()
    assert be.advances == 150                   # hasta 15 s simulados, no indefinido


def test_base_se_reubica_con_el_brazo_quieto():
    be = FakeWorld(xy=(-0.02, 0.0))
    be.run_reset()
    assert all(c.speed <= 0.01 for c in be.robot_calls())
    assert np.linalg.norm(be._base_pose[:2, 3]) == pytest.approx(0.0, abs=1e-9)   # sin deslizamiento tras reubicar
    assert be.reset_error < 0.002               # refleja la posición ya aquietada, no la de la primera convergencia


class Residual(FakeWorld):
    """Tras converger en posición, el brazo mantiene `floor` rad/s durante las primeras `steps` esperas."""

    converge = 6                                # avances que tarda el brazo en entrar en la banda de 0.02 rad

    def __init__(self, floor, steps=10 ** 6):
        self.floor, self.steps = floor, steps
        super().__init__()

    def advance(self, dt):
        super().advance(dt)
        if self.advances < self.converge + self.steps:
            self.speed = max(self.speed, self.floor)
            self.emit_joints()


@pytest.mark.parametrize('floor, ok', [(0.009, True), (0.011, False)])
def test_umbral_de_velocidad(floor, ok):
    be = Residual(floor)
    if ok:
        be.run_reset()
    else:
        with pytest.raises(RuntimeError, match='aquietó'):
            be.run_reset()


@pytest.mark.parametrize('steps, ok', [(10, True), (11, False)])
def test_la_espera_dura_a_lo_sumo_un_segundo(steps, ok):
    be = Residual(0.5, steps)
    if ok:
        be.run_reset()
        assert be.robot_calls()
    else:
        with pytest.raises(RuntimeError, match='aquietó'):
            be.run_reset()
        assert be.advances == be.converge + 10


def test_brazo_en_movimiento_no_reubica_la_base():
    be = Residual(0.5)
    with pytest.raises(RuntimeError, match='aquietó'):
        be.run_reset()
    assert not be.robot_calls()


def test_torso_en_movimiento_no_reubica_la_base():
    be = FakeWorld()
    be.lift_speed = 0.5
    with pytest.raises(RuntimeError, match='aquietó'):
        be.run_reset()
    assert not be.robot_calls()


def test_sin_velocidades_en_joint_states_no_reubica_la_base():
    be = FakeWorld()
    be.joint_speeds = False
    with pytest.raises(RuntimeError, match='aquietó'):
        be.run_reset()
    assert not be.robot_calls()


def test_posicion_perdida_durante_la_espera_no_reubica_la_base():
    class Empujado(FakeWorld):
        def advance(self, dt):
            super().advance(dt)
            if self.advances >= 7:              # ya en la banda de 0.02 rad: algo lo saca y queda quieto
                self.target = self.q = np.full(6, 0.05)
                self.speed = 0.0
                self.emit_joints()

    be = Empujado()
    with pytest.raises(RuntimeError, match='aquietó'):
        be.run_reset()
    assert not be.robot_calls()


class Congelada(FakeWorld):
    """El tópico de pose deja de publicar al reubicar: el reloj avanza y la pose guardada queda vieja."""

    def publish_pose(self):
        if not self.robot_calls():
            self.emit_pose()


def test_pose_vieja_no_se_toma_por_verificada():
    be = Congelada(slide=0.0)                   # la pose vieja parece correcta: solo la delata su sello de tiempo
    with pytest.raises(TimeoutError, match='pose reciente'):
        be.run_reset()
    assert len(be.robot_calls()) == 1


def test_una_muestra_perdida_no_aborta_el_reset():
    class MuestraPerdida(FakeWorld):
        dropped = False

        def publish_pose(self):
            if self.robot_calls() and not self.dropped:
                self.dropped = True
                return
            self.emit_pose()

    be = MuestraPerdida(xy=(-0.008, 0.0))
    be.run_reset()
    assert be.dropped and np.linalg.norm(be._base_pose[:2, 3]) <= 0.005


@pytest.mark.parametrize('age, ok', [(0.015, True), (0.03, False)])
def test_antiguedad_maxima_de_la_pose(age, ok):
    class Atrasada(FakeWorld):
        def publish_pose(self):
            self.emit_pose(age=age if self.robot_calls() else 0.0)

    be = Atrasada(slide=0.0)
    if ok:
        be.run_reset()
    else:
        with pytest.raises(TimeoutError, match='pose reciente'):
            be.run_reset()


def test_servicio_sin_respuesta_propaga_el_timeout():
    class SinServicio(FakeWorld):
        def on_set_pose(self, c):
            if c.name == 'robot_v3':
                raise TimeoutError('Tiempo de espera agotado: set_pose')

    with pytest.raises(TimeoutError, match='set_pose'):
        SinServicio().run_reset()


@pytest.mark.parametrize('offset, ok', [((0.0049, 0.0), True), ((0.0051, 0.0), False),
                                        ((0.0, np.radians(0.49)), True), ((0.0, np.radians(0.51)), False)])
def test_tolerancias_de_xy_y_yaw(offset, ok):
    be = FakeWorld(offset=offset)
    if ok:
        be.run_reset()
    else:
        with pytest.raises(RuntimeError, match='base'):
            be.run_reset()


@pytest.mark.parametrize('tilt, ok', [(0.49, True), (0.51, False)])
def test_tolerancia_de_inclinacion(tilt, ok):
    be = FakeWorld(tilt=(np.radians(tilt), 0.0), rest_tilt=(np.radians(tilt), 0.0))
    if ok:
        be.run_reset()
    else:
        with pytest.raises(RuntimeError, match='base'):
            be.run_reset()
