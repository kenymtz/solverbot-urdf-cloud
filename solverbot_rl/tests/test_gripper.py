"""Pruebas de la pinza: precierre, rampa, detección de dedos detenidos, acoplamiento emulado y orientaciones."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solverbot_rl import task as T  # noqa: E402
from solverbot_rl.config import load_config  # noqa: E402
from solverbot_rl.envs.pick_place import SolverBotPickPlaceEnv  # noqa: E402
from solverbot_rl.expert import grasp_rotations  # noqa: E402
from solverbot_rl.gripper import coupled_finger_targets, is_squeezing  # noqa: E402

CFG = load_config()


# ------------------------------------------------------------------------- acoplamiento emulado (solverbot_rl.gripper)
def test_free_motion_is_mirrored():
    t7, t8 = coupled_finger_targets(0.032, 0.032, -0.032)
    assert (t7, t8) == (0.032, -0.032)
    t7, t8 = coupled_finger_targets(0.025, 0.0265, -0.0265)     # cierre libre con retardo de seguimiento < squeeze
    assert np.isclose(t7, -t8)


def test_symmetric_squeeze_pushes_both_inward():
    t7, t8 = coupled_finger_targets(0.0, 0.020, -0.020, squeeze=0.002)
    assert np.isclose(t7, 0.018) and np.isclose(t8, -0.018)


def test_block_drift_is_corrected():
    """Par desplazado hacia joint8 (j7 = 15, j8 = -25 mm): j7 debe abrir y j8 cerrar (fuerza restauradora)."""
    q7, q8 = 0.015, -0.025
    t7, t8 = coupled_finger_targets(0.0, q7, q8, squeeze=0.002)
    assert t7 > q7 and t8 > q8
    # y al revés
    t7, t8 = coupled_finger_targets(0.0, 0.025, -0.015, squeeze=0.002)
    assert t7 < 0.025 and t8 < -0.015


def test_targets_respect_joint_limits():
    t7, t8 = coupled_finger_targets(0.0, 0.034, -0.002, squeeze=0.002)
    assert 0.0 <= t7 <= 0.035 and -0.035 <= t8 <= 0.0


def test_not_squeezing_when_opening():
    assert not is_squeezing(0.032, 0.020, -0.020, 0.002)
    assert is_squeezing(0.0, 0.020, -0.020, 0.002)


# ------------------------------------------------------------------------- precierre y rampa en el entorno
def test_open_command_never_reaches_joint_limit():
    assert CFG['robot']['gripper_open'] <= 0.033      # informe WSL: bloqueo de joint8 al cerrar desde 35 mm
    env = SolverBotPickPlaceEnv(backend='kinematic')
    env.reset(seed=0)
    for _ in range(3):
        _, _, _, _, info = env.step(np.array([0, 0, 0, 0, 0, 0, -1], np.float32))
    assert np.isclose(info['gripper_target'], CFG['robot']['gripper_open'])


def test_gripper_command_is_rate_limited():
    env = SolverBotPickPlaceEnv(backend='kinematic')
    env.reset(seed=0)
    prev = CFG['robot']['gripper_open']
    rate = CFG['robot']['gripper_rate']
    for _ in range(5):
        _, _, _, _, info = env.step(np.array([0, 0, 0, 0, 0, 0, 1], np.float32))
        assert prev - info['gripper_target'] <= rate + 1e-9
        prev = info['gripper_target']


# ------------------------------------------------------------------------- detección de dedos detenidos
def _s(g, target, near=True, q8=None):
    obj = np.array([0.2, 0.5, 0.76])
    return T.TaskState(tcp_pos=obj if near else obj + [0.1, 0, 0], tcp_z=np.array([0, 0, -1.0]), obj_pos=obj,
                       gripper_q=g, gripper_cmd_closed=True, arm_points=np.zeros((1, 3)) + [0, 0, 1.0],
                       gripper_target=target, gripper_q8=q8)


def test_fingers_following_target_are_not_a_grasp():
    """Sin objeto los dedos siguen al objetivo (A/B del informe): no es agarre aunque la apertura sea ~20 mm."""
    p = T.TaskParams.from_config(CFG, 'grasp')
    assert not T.fingers_blocked_by_object(_s(0.020, 0.0195), p)


def test_fingers_stopped_above_target_near_object_is_grasp():
    p = T.TaskParams.from_config(CFG, 'grasp')
    assert T.fingers_blocked_by_object(_s(0.020, 0.010), p)
    assert not T.fingers_blocked_by_object(_s(0.020, 0.010, near=False), p)


def test_mimic_error():
    assert np.isclose(T.mimic_error(_s(0.020, 0.0, q8=-0.035)), 0.015)
    assert T.mimic_error(_s(0.020, 0.0, q8=None)) == 0.0


# ------------------------------------------------------------------------- orientaciones de agarre
def test_grasp_rotations_are_proper_and_face_aligned():
    yaw = 0.3
    rots = grasp_rotations(yaw)
    assert rots[0][0] == 0                                   # primero la más vertical
    for tilt, r in rots:
        assert np.allclose(r.T @ r, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(r), 1.0)
        c = r[:, 1]                                          # eje de cierre: horizontal y perpendicular a una cara
        assert abs(c[2]) < 1e-9
        ang = (np.arctan2(c[1], c[0]) - yaw) % (np.pi / 2)
        assert min(ang, np.pi / 2 - ang) < 1e-9
        assert np.isclose(np.degrees(np.arccos(-r[2, 2])), tilt)


def test_descent_reaches_planned_depth_kinematic():
    """Regresión: el descenso no debe cortarse antes de la profundidad planificada (bug de timeout que se acortaba
    al consumir waypoints). Backend cinemático, sin física."""
    from solverbot_rl.expert import ScriptedExpert
    env = SolverBotPickPlaceEnv(backend='kinematic', level=4, overrides={'observation.object_source': 'privileged'})
    ex = ScriptedExpert(env)
    env.reset(seed=40005)
    ex.reset()
    for _ in range(120):
        env.step(ex.act())
        if ex.phase == 'close':
            break
    s = env.unwrapped._state
    target_rel = CFG['task']['grasp_dz']
    assert ex.phase == 'close'
    assert abs((s.tcp_pos[2] - s.obj_pos[2]) - target_rel) < 0.008
