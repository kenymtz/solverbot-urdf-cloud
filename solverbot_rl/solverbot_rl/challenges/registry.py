"""Registro de desafíos: nombre corto -> entorno, experto de referencia e id de Gymnasium."""
from __future__ import annotations

import importlib

CHALLENGES = {
    'hri': {'gym_id': 'SolverBotHRI-v0', 'env': 'solverbot_rl.challenges.hri_env:HRIEnv',
            'expert': 'solverbot_rl.challenges.hri_env:HRIExpert', 'levels': 'solverbot_rl.challenges.hri_env:LEVELS'},
    'gpsr': {'gym_id': 'SolverBotGPSR-v0', 'env': 'solverbot_rl.challenges.gpsr_env:GPSREnv',
             'expert': 'solverbot_rl.challenges.gpsr_env:GPSRPlanner',
             'levels': 'solverbot_rl.challenges.gpsr_env:LEVELS'},
    'laundry': {'gym_id': 'SolverBotLaundry-v0', 'env': 'solverbot_rl.challenges.laundry_env:LaundryEnv',
                'expert': 'solverbot_rl.challenges.laundry_env:LaundryExpert',
                'levels': 'solverbot_rl.challenges.laundry_env:LEVELS'},
}


def _load(path):
    mod, name = path.split(':')
    return getattr(importlib.import_module(mod), name)


def make_env(challenge: str, **kw):
    return _load(CHALLENGES[challenge]['env'])(**kw)


def make_expert(challenge: str, env):
    """Experto GUIONADO de referencia (no es una política aprendida)."""
    return _load(CHALLENGES[challenge]['expert'])(env.unwrapped)


def levels(challenge: str) -> list:
    return _load(CHALLENGES[challenge]['levels'])


def register_all():
    from gymnasium.envs.registration import register, registry
    for c in CHALLENGES.values():
        if c['gym_id'] not in registry:
            register(id=c['gym_id'], entry_point=c['env'], disable_env_checker=True)
