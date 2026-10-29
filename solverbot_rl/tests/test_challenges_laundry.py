"""Doing Laundry (etapa intermedia, prendas = proxies rígidos): interfaz Gymnasium, visibilidad y experto."""
import numpy as np
import pytest

from solverbot_rl.challenges.laundry_env import SKILLS, TARGETS, LaundryEnv, LaundryExpert


@pytest.fixture(scope='module')
def env():
    e = LaundryEnv(level=3)
    yield e
    e.close()


def test_spaces_and_proxy_label(env):
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs) and np.all(np.isfinite(obs))
    assert tuple(env.action_space.nvec) == (len(SKILLS), len(TARGETS))
    assert info['garment_model'] == 'rigid_proxy'
    for g in env.garments:                      # prendas dentro del cesto
        x, y, z = env.objects_truth[g]
        assert abs(x - env.basket_xy[0]) < env.basket_size[0] / 2 and abs(y - env.basket_xy[1]) < env.basket_size[1] / 2
        assert z < env.basket_size[2]


def test_reset_reproducible(env):
    env.reset(seed=9)
    a = dict(env.objects_truth)
    env.reset(seed=9)
    assert a == env.objects_truth


def test_head_camera_cannot_see_basket_bottom_from_afar(env):
    env.reset(seed=2)
    # robot en la entrada: la prenda del fondo del cesto queda tapada por el borde
    for g in env.garments:
        assert g not in env.perceivable_objects_truth()


def test_pregrasp_requires_pose(env):
    env.reset(seed=3)
    _, r, _, _, info = env.step(np.array([SKILLS.index('pregrasp'), 0]))
    assert not info['skill_ok'] and r < 0 and not info['is_success']


def test_expert_level3():
    e = LaundryEnv(level=3)
    obs, _ = e.reset(seed=1000)
    ex = LaundryExpert(e)
    done = False
    while not done:
        obs, r, te, tr, info = e.step(ex.act(obs))
        done = te or tr
    s = info['episode_summary']
    assert s['success'] and not s['contact'], s
    assert s['garment_model'] == 'rigid_proxy'
    e.close()
