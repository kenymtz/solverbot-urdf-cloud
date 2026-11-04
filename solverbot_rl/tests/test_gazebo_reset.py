"""GazeboBackend.reset() contra un mundo falso (sin ROS): orden de operaciones y verificación de la pose de la base."""
import numpy as np
import pytest

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


def matrix(xyz, q):
    x, y, z, w = q
    m = np.eye(4)
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    m[:3, 3] = xyz
    return m


class FakeWorld(GazeboBackend):
    """Reproduce lo medido: el regreso del brazo desliza la base, un teletransporte plano aterriza con residuo de yaw y
    las poses solo se publican cuando avanza la simulación."""

    def __init__(self, xy=(0.0, 0.0), yaw=0.0, tilt=(ROLL, PITCH), obeys_set_pose=True):
        self.obj_name, self.obj_half_h, self.table_h, self.arm_joints = 'obj', 0.02, 0.4, JOINTS
        self.timeout = 0.2
        self.base_max_age, self.base_tol = 0.02, (0.005, np.radians(0.5), np.radians(0.5))
        self._clock, self._base_t = 10.0, 10.0
        self._js = {j: 1.0 for j in JOINTS} | {'torso_lift_joint': 0.0}
        self.arm_target = np.ones(6)            # el brazo sostiene la pose final del episodio anterior
        self.xyz, self.tilt, self.yaw = np.array([*xy, REST_Z]), tilt, yaw
        self.obeys_set_pose = obeys_set_pose
        self.landing = None                     # pose pendiente de un teletransporte (visible al avanzar)
        self.log = []
        self.obj_table_base = None
        self._publish()

    def _publish(self):
        self._base_q = quat(*self.tilt, self.yaw)
        self._base_pose = matrix(self.xyz, self._base_q)
        self._base_t = self._clock

    def _pub(self, pub, values):
        pass

    pub_lift = pub_head = None

    def command(self, arm_q, gripper):
        self.arm_target = np.asarray(arm_q, float)

    def advance(self, dt):
        self._clock += dt
        q = np.array([self._js[j] for j in JOINTS])
        moved = float(np.max(np.abs(q - self.arm_target)))
        if moved > 0.02:                        # el brazo en movimiento desliza la base
            self.xyz[0] -= 0.01
            self.yaw += np.radians(2.0)
        for j, t in zip(JOINTS, self.arm_target):
            self._js[j] = t if moved < 0.3 else self._js[j] + (t - self._js[j]) * 0.5
        if self.landing is not None:
            self.xyz, self.yaw, self.tilt = self.landing
            self.landing = None
        self._publish()

    def set_pose(self, name, xyz, yaw=0.0, quat=None):
        q = np.array([self._js[j] for j in JOINTS])
        self.log.append((name, float(np.max(np.abs(q - HOME))), xyz, quat))
        if name == self.obj_name and xyz[2] > self.table_h:
            self.obj_table_base = self._base_pose.copy()
        if name == 'robot_v3' and self.obeys_set_pose:
            if quat is None:                    # plano: se reacomoda a la inclinación de reposo y gira
                self.landing = (np.array([xyz[0] - 0.002, xyz[1] + 0.0016, REST_Z]), np.radians(0.7), (ROLL, PITCH))
            else:
                self.landing = (np.array(xyz, float), yaw_of(quat), tilt_of(quat))
        return True

    def robot_calls(self):
        return [c for c in self.log if c[0] == 'robot_v3']

    def run_reset(self):
        self.reset(None, [0.3, 0.0], 0.0, HOME, 0.03, 0.0, HEAD)


def yaw_of(q):
    x, y, z, w = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def tilt_of(q):
    x, y, z, w = q
    return (float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))), float(np.arcsin(2 * (w * y - z * x))))


def test_base_se_reubica_despues_de_que_el_brazo_converge():
    be = FakeWorld()
    be.run_reset()
    assert be.robot_calls(), 'la base no se reubicó'
    assert all(err < 0.02 for _, err, _, _ in be.robot_calls())
    assert np.linalg.norm(be._base_pose[:2, 3]) <= 0.005
    assert abs(be.yaw) <= np.radians(0.5)


def test_reubicacion_conserva_altura_e_inclinacion_de_reposo():
    be = FakeWorld(xy=(-0.018, -0.003), yaw=np.radians(3.8))
    be.run_reset()
    _, _, xyz, q = be.robot_calls()[-1]
    assert xyz[:2] == [0.0, 0.0]
    assert xyz[2] == pytest.approx(REST_Z)
    assert yaw_of(q) == pytest.approx(0.0, abs=1e-9)
    assert tilt_of(q) == pytest.approx((ROLL, PITCH), abs=1e-6)


def test_exito_del_servicio_no_es_pose_verificada():
    be = FakeWorld(xy=(-0.02, 0.0), obeys_set_pose=False)
    with pytest.raises(RuntimeError, match='base'):
        be.run_reset()
    assert 1 <= len(be.robot_calls()) <= 3      # intentos acotados


def test_base_volcada_cae_plana_y_luego_corrige_el_yaw():
    be = FakeWorld(tilt=(np.radians(20.0), 0.0))
    be.run_reset()
    calls = be.robot_calls()
    assert calls[0][3] is None                  # plano: no se conserva una inclinación anómala
    assert calls[-1][3] is not None
    assert abs(be.yaw) <= np.radians(0.5) and abs(be.tilt[0]) <= np.radians(0.5)


def test_base_volcada_se_endereza_antes_de_mover_el_brazo():
    class BrazoTrabado(FakeWorld):
        """Volcada, el brazo no puede volver: solo converge cuando la base ya está derecha."""

        def advance(self, dt):
            held = abs(self.tilt[0]) > np.radians(0.5)
            js = dict(self._js)
            super().advance(dt)
            if held:
                self._js = js

    be = BrazoTrabado(tilt=(np.radians(20.0), 0.0))
    be.run_reset()
    assert be.robot_calls()[0][1] > 0.02        # se enderezó con el brazo todavía lejos de la pose inicial
    assert abs(be.tilt[0]) <= np.radians(0.5)


def test_objeto_se_ubica_respecto_de_la_base_ya_reubicada():
    be = FakeWorld(xy=(-0.02, 0.0), yaw=np.radians(3.0))
    be.run_reset()
    assert np.linalg.norm(be.obj_table_base[:2, 3]) <= 0.005


def test_error_si_la_base_se_mueve_despues_de_verificarla():
    class Empujada(FakeWorld):
        def set_pose(self, name, xyz, yaw=0.0, quat=None):
            ok = super().set_pose(name, xyz, yaw, quat)
            if name == self.obj_name and xyz[2] > self.table_h:
                self.xyz[0] -= 0.02             # algo la empuja al apoyar el objeto
            return ok

    with pytest.raises(RuntimeError, match='base'):
        Empujada().run_reset()
