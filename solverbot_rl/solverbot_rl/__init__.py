"""solverbot_rl: entorno Gymnasium de Pick and Place para SolverBot v3 (solo simulación)."""
from gymnasium.envs.registration import register

register(id='SolverBotPickPlace-v0', entry_point='solverbot_rl.envs.pick_place:SolverBotPickPlaceEnv')
