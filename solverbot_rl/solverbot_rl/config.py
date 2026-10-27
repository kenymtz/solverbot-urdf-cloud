"""Carga de la configuración de la tarea (solverbot_sim/config/pick_place.yaml) y del URDF."""
from __future__ import annotations

import copy
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_CONFIG = os.path.join(REPO, 'solverbot_sim', 'config', 'pick_place.yaml')
# URDF de robot_v3 (xacro ros2_control:=none) versionado para el backend cinemático y las pruebas sin ROS.
# Regenerar: xacro robot_v3_description/urdf/robot_v3.urdf.xacro > solverbot_rl/solverbot_rl/data/robot_v3.urdf
DEFAULT_URDF = os.path.join(HERE, 'data', 'robot_v3.urdf')


def load_config(path: str | None = None, overrides: dict | None = None) -> dict:
    path = path or os.environ.get('SOLVERBOT_RL_CONFIG', DEFAULT_CONFIG)
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    for dotted, value in (overrides or {}).items():
        node = cfg
        *keys, last = dotted.split('.')
        for k in keys:
            node = node[k]
        node[last] = copy.deepcopy(value)
    return cfg


def load_urdf(path: str | None = None) -> str:
    with open(path or DEFAULT_URDF) as fh:
        return fh.read()
