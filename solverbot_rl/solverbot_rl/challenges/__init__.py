"""Desafíos HRI, GPSR y Doing Laundry como entornos Gymnasium (ver docs/DESAFIOS_SPEC.md).

Registro en Gymnasium (import perezoso; el mapa se construye al crear el primer entorno):
    gym.make('SolverBotHRI-v0', backend='kinematic', level=0)
"""
from .registry import CHALLENGES, make_env, make_expert, register_all

register_all()

__all__ = ['CHALLENGES', 'make_env', 'make_expert', 'register_all']
