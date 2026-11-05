"""Etiquetador de IK recalculada en cada paso (imitation/dynamic_label_expert.py, Fase C0).

Fija: preferencia por B sin compromiso, continuidad con A cuando la muñeca medida ya va por A, joint6 sin girar
(objetivo = joint6 medido), objetivo que es función del estado medido (mismo estado -> mismo objetivo), robustez a
ruido articular pequeño, ausencia de saltos de objetivo en la frontera singular de la muñeca, penalización de límites
solo en el coste de selección, y que ScriptedExpert y BranchConsistentLabelExpert siguen intactos.
"""
import numpy as np
import pytest

from solverbot_rl.envs.pick_place import SolverBotPickPlaceEnv
from solverbot_rl.imitation import branch_analysis as BA
from solverbot_rl.imitation.dynamic_label_expert import MeasuredIKLabelExpert
from solverbot_rl.imitation.experiment import env_kwargs, level_index, load_experiment
from solverbot_rl.imitation.rollout import run_episode
from solverbot_rl.imitation.start_override import override_start_j4


@pytest.fixture(scope='module')
def env():
    e = SolverBotPickPlaceEnv(backend='kinematic', **env_kwargs(load_experiment('reach_bc_v1')))
    yield e
    e.close()


def _target(env):
    u = env.unwrapped
    return u.backend.object_pose_truth() + np.array([0, 0, u.params.pregrasp_height])


def _set_q(env, q):
    u = env.unwrapped
    u.backend.reset(np.random.default_rng(0), u.start_obj_xy, 0.0, q, u.cfg['robot']['gripper_open'],
                    u.cfg['robot']['lift'], u.cfg['robot']['head'])
    u._q_cmd = np.asarray(q, float).copy()


def test_prefers_B_from_home_and_keeps_j6(env):
    ex = MeasuredIKLabelExpert(env)
    env.reset(seed=916000, options={'level': level_index(env, 'reach')})
    ex.reset()
    q0 = ex._q()
    qg = ex.choose(_target(env))
    assert ex.family == 'B' and ex.decision == 'preferencia'
    assert abs(qg[5] - q0[5]) < 1e-9                       # joint6 del objetivo = joint6 medido


def test_goal_is_function_of_measured_state(env):
    """Dos etiquetadores con historia distinta, en el MISMO estado medido, dan el mismo objetivo."""
    env.reset(seed=916000, options={'level': level_index(env, 'reach')})
    p = _target(env)
    a = MeasuredIKLabelExpert(env)
    a.reset()
    for _ in range(6):
        env.step(a.act())
    q_mid = a._q()
    g_hist = a.choose(p)
    b = MeasuredIKLabelExpert(env)
    b.reset()
    _set_q(env, q_mid)
    g_fresh = b.choose(p)
    assert np.linalg.norm(g_hist - g_fresh) < 0.02
    assert a.family == b.family


def test_continuity_keeps_A_when_wrist_committed(env):
    ex = MeasuredIKLabelExpert(env)
    env.reset(seed=916000, options={'level': level_index(env, 'reach')})
    u = env.unwrapped
    p = _target(env)
    ex.reset()
    rep_a = [s['q'] for s in BA.ik_solution_set(env, np.array(u.start_q), p, n_starts=6, tilts=(20,))
             if s['family'] == 'A'][0]
    _set_q(env, 0.3 * np.array(u.start_q) + 0.7 * rep_a)
    ex.reset()
    qg = ex.choose(p)
    assert ex.family == 'A' and ex.decision == 'continuidad' and BA.family_of(qg) == 'A'


def test_small_joint_noise_does_not_change_solution(env):
    ex = MeasuredIKLabelExpert(env)
    env.reset(seed=916410, options={'level': level_index(env, 'reach_var_m')})
    p = _target(env)
    ex.reset()
    q = ex._q()
    g0 = ex.choose(p)
    fam0, tilt0 = ex.family, ex.tilt
    rng = np.random.default_rng(0)
    for _ in range(4):
        _set_q(env, q + rng.normal(0, 0.005, 6))
        ex.reset()
        g = ex.choose(p)
        assert (ex.family, ex.tilt) == (fam0, tilt0)
        assert np.linalg.norm(g[:5] - g0[:5]) < 0.05


@pytest.mark.parametrize('level,seed,j4', [('reach', 916010, None), ('reach_var_l', 916610, None),
                                           ('reach_var_m', 940401, 0.5)])
def test_executes_without_goal_jumps(level, seed, j4):
    """Incluye 940401 con joint4 inicial 0.5: objetivo en la frontera singular (|joint5| ~ 0.05), donde antes el
    objetivo alternaba entre soluciones del continuo de joint4 (saltos de 0.4 rad)."""
    e = SolverBotPickPlaceEnv(backend='kinematic', **env_kwargs(load_experiment('reach_bc_v1')))
    if j4 is not None:
        override_start_j4(e, value=j4)
    ex = MeasuredIKLabelExpert(e)
    arrays, s = run_episode(e, seed, level_index(e, level), expert=ex)
    jumps = [d['goal_jump'] for d in ex.decisions if d.get('goal_jump') is not None]
    assert s['success'] and not s['collision']
    assert max(jumps, default=0.0) < 0.1
    assert len({d['family'] for d in ex.decisions}) == 1
    assert np.all(np.abs(arrays['actions']) <= 1.0)
    e.close()


def test_limit_penalty_only_in_selection_cost(env):
    ex = MeasuredIKLabelExpert(env)
    u = env.unwrapped
    q = np.array(u.start_q, float)
    near = q.copy()
    near[4] = u.hi[4] - 0.05
    assert ex.limit_penalty(near) > ex.limit_penalty(q)
    assert ex.cost(q, near) > float(np.linalg.norm((near - q)[:5]))


def test_previous_labelers_untouched():
    from solverbot_rl.expert import ScriptedExpert
    from solverbot_rl.imitation.branch_label_expert import BranchConsistentLabelExpert
    assert not issubclass(ScriptedExpert, MeasuredIKLabelExpert)
    assert not issubclass(BranchConsistentLabelExpert, MeasuredIKLabelExpert)
    assert ScriptedExpert.act is not MeasuredIKLabelExpert.act
