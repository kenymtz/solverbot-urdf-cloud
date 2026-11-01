"""Regresión de Pick and Place frente a los cambios compartidos de la rama de desafíos (PR #4).

Cambios compartidos con P&P y por qué no lo afectan (ver DESAFIOS_SPEC.md, "Auditoría de integración"):
  * controllers_rl.yaml: SOLO se agregaron límites de aceleración al diff_drive_controller. P&P no comanda la base
    (la orden de la base es siempre 0), y un límite de aceleración solo actúa cuando la velocidad comandada cambia.
  * make_worlds.py / home_arena.sdf: cambios de mobiliario del mundo de la casa. P&P usa pick_place_training.sdf,
    que no cambia.
Estos tests fijan esas condiciones para que un cambio futuro que las rompa falle aquí.
"""
import os
import re
import subprocess
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIM = os.path.join(REPO, 'solverbot_sim')
RL = os.path.join(REPO, 'solverbot_rl', 'solverbot_rl')

# Parámetros de los controladores que usa P&P, tal como están en fix/solverbot-grasp-headless (04a0dbc)
PICK_PLACE_CONTROLLERS = {
    'arm_position_controller': {'joints': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
                                'interface_name': 'position'},
    'gripper_position_controller': {'joints': ['joint7', 'joint8'], 'interface_name': 'position'},
    'lift_position_controller': {'joints': ['torso_lift_joint'], 'interface_name': 'position'},
    'head_position_controller': {'joints': ['pan_joint', 'tilt_joint'], 'interface_name': 'position'},
}
DIFF_DRIVE_BASE = {'left_wheel_names': ['left_wheel_joint'], 'right_wheel_names': ['right_wheel_joint'],
                   'wheel_separation': 0.5621, 'wheel_radius': 0.0829, 'publish_rate': 50.0, 'odom_frame_id': 'odom',
                   'base_frame_id': 'base_footprint', 'enable_odom_tf': True, 'cmd_vel_timeout': 0.5,
                   'linear.x.has_velocity_limits': True, 'linear.x.max_velocity': 0.4, 'linear.x.min_velocity': -0.4,
                   'angular.z.has_velocity_limits': True, 'angular.z.max_velocity': 1.0,
                   'angular.z.min_velocity': -1.0}
ADDED_KEYS = {'linear.x.has_acceleration_limits', 'linear.x.max_acceleration', 'linear.x.min_acceleration',
              'angular.z.has_acceleration_limits', 'angular.z.max_acceleration', 'angular.z.min_acceleration'}


def _controllers():
    with open(os.path.join(SIM, 'config', 'controllers_rl.yaml')) as fh:
        return yaml.safe_load(fh)


def test_pick_place_controllers_unchanged():
    c = _controllers()
    assert c['controller_manager']['ros__parameters']['update_rate'] == 100
    for name, params in PICK_PLACE_CONTROLLERS.items():
        assert c[name]['ros__parameters'] == params, name


def test_diff_drive_only_gains_acceleration_limits():
    p = _controllers()['diff_drive_controller']['ros__parameters']
    for k, v in DIFF_DRIVE_BASE.items():
        assert p[k] == v, k                                # nada de lo previo cambió
    assert set(p) - set(DIFF_DRIVE_BASE) == ADDED_KEYS     # solo se agregaron los límites de aceleración
    assert p['linear.x.max_acceleration'] == 0.4 and p['angular.z.max_acceleration'] == 1.2


def test_pick_place_code_never_commands_the_base():
    """El límite de aceleración no puede afectar a P&P si P&P nunca publica una velocidad de base."""
    files = [os.path.join(RL, 'envs', 'pick_place.py'), os.path.join(RL, 'backends', 'gazebo.py'),
             os.path.join(RL, 'backends', 'kinematic.py'), os.path.join(RL, 'expert.py'),
             os.path.join(RL, 'gripper.py'), os.path.join(RL, 'task.py')]
    for f in files:
        if not os.path.exists(f):
            continue
        text = open(f).read()
        assert not re.search(r'cmd_vel|diff_drive|Twist', text), f


def test_pick_place_training_world_is_reproducible_and_clean():
    """El mundo de P&P se regenera idéntico desde make_worlds.py y no contiene entidades de los desafíos."""
    path = os.path.join(SIM, 'worlds', 'pick_place_training.sdf')
    committed = open(path).read()
    out = subprocess.run([sys.executable, '-c',
                          'import sys, yaml, os; sys.path.insert(0, "tools"); import make_worlds as mw; '
                          'task = yaml.safe_load(open("config/pick_place.yaml"))["world"]; '
                          'sys.stdout.write(mw.training_world(task))'],
                         cwd=SIM, capture_output=True, text=True, check=True).stdout
    assert out == committed
    for name in ('person_a', 'entrance_door_leaf', 'laundry_basket', 'tshirt_1', 'laundry_door_leaf'):
        assert name not in committed
