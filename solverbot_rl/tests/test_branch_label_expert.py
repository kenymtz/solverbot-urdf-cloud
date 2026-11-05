"""Etiquetador con rama de muñeca consistente (imitation/branch_label_expert.py, Fase B1).

Fija: preferencia por B cuando la muñeca no está comprometida, continuidad cuando ya lo está (sin forzar el regreso),
ningún cambio de familia al ejecutar por sí mismo, y que ScriptedExpert sigue siendo el experto por defecto.
"""
import numpy as np
import pytest

from solverbot_rl.envs.pick_place import SolverBotPickPlaceEnv
from solverbot_rl.imitation import branch_analysis as BA
from solverbot_rl.imitation.branch_label_expert import BranchConsistentLabelExpert
from solverbot_rl.imitation.experiment import env_kwargs, level_index, load_experiment
from solverbot_rl.imitation.rollout import run_episode


@pytest.fixture(scope='module')
def env():
    e = SolverBotPickPlaceEnv(backend='kinematic', **env_kwargs(load_experiment('reach_bc_v1')))
    yield e
    e.close()


def test_prefers_B_when_uncommitted(env):
    ex = BranchConsistentLabelExpert(env)
    env.reset(seed=916000, options={'level': level_index(env, 'reach')})
    ex.reset()
    p = env.unwrapped.backend.object_pose_truth() + np.array([0, 0, env.unwrapped.params.pregrasp_height])
    q = ex._ik(p)
    assert ex.family == 'B' and ex.decision == 'preferencia' and BA.family_of(q) == 'B'


def test_continuity_keeps_A_when_wrist_already_committed(env):
    """Estado de DAgger: la muñeca medida ya va por A. La etiqueta sigue A (no fuerza la rotación de regreso a B)."""
    ex = BranchConsistentLabelExpert(env)
    env.reset(seed=916000, options={'level': level_index(env, 'reach')})
    u = env.unwrapped
    p = u.backend.object_pose_truth() + np.array([0, 0, u.params.pregrasp_height])
    ex.reset()
    best = ex._solutions(np.array([u._joints()[j] for j in u.arm_joints]), p)
    rep_a = best[('A', 20)]
    q_mid = 0.3 * np.array(u.start_q) + 0.7 * rep_a          # 70 % del camino hacia la solución A
    u.backend.reset(np.random.default_rng(0), u.start_obj_xy, 0.0, q_mid, u.cfg['robot']['gripper_open'],
                    u.cfg['robot']['lift'], u.cfg['robot']['head'])
    u._q_cmd = q_mid.copy()
    ex.reset()
    q = ex._ik(u.backend.object_pose_truth() + np.array([0, 0, u.params.pregrasp_height]))
    assert ex.decision == 'continuidad' and ex.family == 'A' and BA.family_of(q) == 'A'


@pytest.mark.parametrize('level,seed', [('reach', 916010), ('reach_var_m', 916410), ('reach_var_l', 916610)])
def test_executes_without_family_changes(env, level, seed):
    ex = BranchConsistentLabelExpert(env)
    arrays, s = run_episode(env, seed, level_index(env, level), expert=ex)
    fams = [d['family'] for d in ex.decisions]
    assert s['success']
    assert len(set(f for f in fams if f is not None)) == 1
    assert np.all(np.abs(arrays['actions']) <= 1.0)


def test_scripted_expert_is_still_the_default():
    from solverbot_rl.expert import ScriptedExpert
    assert not issubclass(ScriptedExpert, BranchConsistentLabelExpert)
    assert ScriptedExpert._ik is not BranchConsistentLabelExpert._ik


def test_regression_121128_goal_is_the_verified_solution(env):
    """Regresión: B@30 era viable pero la búsqueda barata del objetivo no la encontraba y el etiquetador caía a la IK
    original (familia A) sin registrarlo. Ahora el objetivo es la solución verificada y la familia no cambia."""
    ex = BranchConsistentLabelExpert(env)
    arrays, s = run_episode(env, 121128, level_index(env, 'reach_var_m'), expert=ex)
    assert ex.family == 'B' and ex.tilt == 30 and ex.decision == 'preferencia'
    assert {d['family'] for d in ex.decisions} == {'B'}
    assert s['success']
