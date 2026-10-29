"""Entorno HRI (backend cinemático): interfaz Gymnasium, reproducibilidad, separación verdad/observación y experto."""
import numpy as np
import pytest

from solverbot_rl.challenges.hri_env import LEVELS, SKILLS, HRIEnv, HRIExpert


@pytest.fixture(scope='module')
def env():
    e = HRIEnv(level=0)
    yield e
    e.close()


def _rollout(env, seed, n_steps, expert=True):
    obs, _ = env.reset(seed=seed)
    ex = HRIExpert(env)
    out = [obs]
    rng = np.random.default_rng(seed)
    for _ in range(n_steps):
        a = ex.act(obs) if expert else rng.integers(env.action_space.nvec)
        obs, r, te, tr, info = env.step(a)
        out.append(obs)
        if te or tr:
            break
    return out, info


def test_spaces(env):
    obs, info = env.reset(seed=3)
    assert env.observation_space.contains(obs)
    assert obs.dtype == np.float32 and np.all(np.isfinite(obs))
    assert tuple(env.action_space.nvec) == (len(SKILLS), 4)
    for k in ('sim_time', 'contact', 'tipped', 'fault', 'backend', 'physical'):
        assert k in info


def test_step_contract_random_actions(env):
    obs, _ = env.reset(seed=5)
    rng = np.random.default_rng(0)
    for _ in range(6):
        a = rng.integers(env.action_space.nvec)
        obs, r, te, tr, info = env.step(a)
        assert env.observation_space.contains(obs)
        assert isinstance(r, float) and np.isfinite(r)
        assert isinstance(te, bool) and isinstance(tr, bool)
        assert {'skill', 'skill_ok', 'terms', 'is_success', 'progress'} <= set(info)
        if te or tr:
            break


def test_reset_reproducible(env):
    a, ia = _rollout(env, 11, 4)
    b, ib = _rollout(env, 11, 4)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, atol=1e-6)
    assert ia['sim_time'] == ib['sim_time']


def test_reset_clears_state(env):
    _rollout(env, 2, 3)
    obs, info = env.reset(seed=2)
    assert info['sim_time'] == 0.0 and not info['contact'] and env.skills_done == 0
    assert env.tracks == {} and env.events == []
    assert all(p.state in ('parked', 'seated', 'walking') for p in env.humans.people.values())


def test_observation_does_not_read_truth(env):
    # mover a una persona en la verdad SIN que el robot la perciba no debe cambiar la observación
    obs, _ = env.reset(seed=4)
    host = env.host
    o1 = env._obs()
    env.humans.people[host].pose[:2] += 0.5
    o2 = env._obs()
    np.testing.assert_array_equal(o1, o2)


def test_invalid_skill_is_penalized_not_rewarded(env):
    env.reset(seed=6)
    _, r, te, tr, info = env.step(np.array([SKILLS.index('follow'), 0]))
    assert not info['skill_ok'] and r < 0 and not info['is_success']


@pytest.mark.parametrize('level', [0, 3])
def test_expert_solves(level):
    e = HRIEnv(level=level)
    _, info = _rollout(e, 1000 + level, 70)
    assert info['episode_summary']['success'], info['episode_summary']
    assert not info['contact'] and not info['tipped']
    e.close()


def test_levels_defined():
    assert [L['name'] for L in LEVELS] == ['recepcion_1', 'acomodar_1', 'acomodar_2', 'presentar', 'cambio_asiento',
                                           'bolsa']
