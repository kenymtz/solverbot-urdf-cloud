"""Carga de experimentos: configuración base SIN cambios + niveles experimentales agregados al final."""
from __future__ import annotations

import os

import yaml

from ..config import load_config

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'experiments')


def experiment_path(name_or_path: str) -> str:
    if os.path.exists(name_or_path):
        return name_or_path
    return os.path.join(EXPERIMENTS, f'{name_or_path}.yaml')


def load_experiment(name_or_path: str) -> dict:
    with open(experiment_path(name_or_path)) as fh:
        exp = yaml.safe_load(fh)
    exp.setdefault('overrides', {})
    exp.setdefault('extra_levels', [])
    return exp


def env_kwargs(exp: dict) -> dict:
    """kwargs para SolverBotPickPlaceEnv: la config base con los niveles originales intactos + los experimentales."""
    base = load_config(exp.get('base_config'))
    names = [lv['name'] for lv in base['curriculum']['levels']]
    for lv in exp['extra_levels']:
        if lv['name'] in names:
            raise ValueError(f'nivel experimental {lv["name"]} duplica un nivel existente')
    overrides = dict(exp['overrides'])
    overrides['curriculum.levels'] = base['curriculum']['levels'] + list(exp['extra_levels'])
    return {'config_path': exp.get('base_config'), 'overrides': overrides}


def level_index(env, level: str | int) -> int:
    """Índice de nivel por nombre ('reach', 'reach_var_s', ...) o número."""
    if isinstance(level, int) or str(level).isdigit():
        return int(level)
    names = [lv['name'] for lv in env.levels]
    if level not in names:
        raise ValueError(f'nivel {level!r} no existe; disponibles: {names}')
    return names.index(level)


def parse_seeds(spec: str) -> list[int]:
    """'100-104,200' -> [100, 101, 102, 103, 104, 200]."""
    out: list[int] = []
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out
