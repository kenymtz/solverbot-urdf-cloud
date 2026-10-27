"""Pruebas del entorno Gymnasium con el backend cinemático (sin simulador, SIN física)."""
import os
import sys

import numpy as np

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import solverbot_rl  # noqa: E402,F401  (registra SolverBotPickPlace-v0)
from solverbot_rl.envs.pick_place import STATE_DIM, SolverBotPickPlaceEnv  # noqa: E402
from solverbot_rl.expert import ScriptedExpert  # noqa: E402


def test_gymnasium_api():
    env = SolverBotPickPlaceEnv(backend='kinematic')
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=0)
    assert obs.shape == (STATE_DIM,) and env.observation_space.contains(obs)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert np.isfinite(r) and 'is_success' in info and 'terms' in info


def test_registered():
    env = gym.make('SolverBotPickPlace-v0', backend='kinematic')
    env.reset(seed=0)
    env.close()


def test_reset_is_reproducible_and_isolated():
    env = SolverBotPickPlaceEnv(backend='kinematic', level=1)

    def rollout(seed):
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        out = [obs]
        for _ in range(10):
            out.append(env.step(rng.uniform(-1, 1, 7).astype(np.float32))[0])
        return np.array(out)
    a = rollout(3)
    rollout(4)                                       # episodio intermedio distinto
    b = rollout(3)
    assert np.allclose(a, b)                         # misma semilla -> mismo episodio, sin contaminación


def test_privileged_info_not_in_observation_by_default():
    env = SolverBotPickPlaceEnv(backend='kinematic')
    assert env.object_source == 'perception'
    _, info = env.reset(seed=0)
    assert info['object_source'] == 'perception'


def test_truncation_at_max_steps():
    env = SolverBotPickPlaceEnv(backend='kinematic', overrides={'control.max_episode_steps': 5,
                                                                 'task.no_progress_steps': 1000})
    env.reset(seed=0)
    for k in range(5):
        _, _, term, trunc, info = env.step(np.zeros(7, np.float32))
    assert trunc and 'episode_summary' in info


def test_scripted_expert_solves_lift():
    """Cota superior: el nivel 'lift' es resoluble y el criterio de éxito lo detecta."""
    env = SolverBotPickPlaceEnv(backend='kinematic', level=3, overrides={'observation.object_source': 'privileged'})
    ex = ScriptedExpert(env)
    env.reset(seed=1)
    ex.reset()
    for _ in range(150):
        _, _, term, trunc, info = env.step(ex.act())
        if term or trunc:
            break
    assert info['is_success'] and info['lifted']
