"""Fugas de información privilegiada: la VERDAD del simulador no puede llegar a la observación ni a la memoria del agente.

Cada habilidad devuelve dos canales: `ok` (verdad: recompensa, métricas, terminación, info) y `belief` (lo que el robot
sabe por sus sensores e interacciones). Antes, `mem['last_ok'] = res.get('belief', res.ok)` caía a la verdad cuando
faltaba `belief`, y algunas memorias observables ('at_basket', 'reached', 'believed_done', 'offered') se escribían con
`ok`. Estos tests fijan la separación, incluso cuando verdad y creencia DIFIEREN.
"""
import ast
import glob
import os

import numpy as np
import pytest

from solverbot_rl.challenges.base_env import MissingBelief, SkillResult
from solverbot_rl.challenges.gpsr_env import GPSREnv
from solverbot_rl.challenges.hri_env import HRIEnv
from solverbot_rl.challenges.laundry_env import LaundryEnv
from solverbot_rl.challenges.nav import foot_center

CH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'solverbot_rl', 'challenges')
ENVS = [HRIEnv, GPSREnv, LaundryEnv]


def _calls(path):
    for node in ast.walk(ast.parse(open(path).read())):
        if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'SkillResult':
            yield node


def test_every_skill_result_declares_belief():
    """Estático: todo SkillResult(...) de los desafíos declara `belief` explícitamente."""
    missing = [f'{os.path.basename(p)}:{n.lineno}' for p in glob.glob(os.path.join(CH, '*.py')) for n in _calls(p)
               if not any(k.arg == 'belief' for k in n.keywords)]
    assert not missing, missing


def test_no_fallback_from_belief_to_truth():
    for p in glob.glob(os.path.join(CH, '*.py')):
        src = open(p).read()
        assert ".get('belief'" not in src, p          # el patrón res.get('belief', res.ok) no puede volver


def test_missing_belief_is_an_error_not_truth():
    with pytest.raises(MissingBelief):
        SkillResult(ok=True).observed()
    assert SkillResult(ok=True, belief=False).observed() is False
    assert SkillResult(ok=False, belief=True).observed() is True


@pytest.mark.parametrize('cls', ENVS)
@pytest.mark.parametrize('ok,belief', [(True, False), (False, True)])
def test_last_ok_follows_belief_when_truth_differs(cls, ok, belief):
    """El paso del entorno escribe la CREENCIA en la memoria/observación aunque la verdad diga lo contrario; la
    verdad sigue llegando a info (métrica) y a la recompensa."""
    env = cls(level=0)
    env.reset(seed=3)
    env._run = lambda skill, arg, terms: SkillResult(ok=ok, belief=belief)
    obs, r, te, tr, info = env.step(np.array([0, 0]))
    assert env.mem['last_ok'] == float(belief)
    assert info['skill_ok'] is ok
    if cls is not GPSREnv:                            # HRI y Laundry exponen last_ok en la observación
        assert obs[-1 if cls is HRIEnv else -3] == pytest.approx(float(belief))
    env.close()


@pytest.mark.parametrize('cls', ENVS)
def test_observation_ignores_truth_only_changes(cls):
    """Mover a las personas / objetos REALES sin que el robot los perciba no cambia la observación."""
    env = cls(level=1)
    env.reset(seed=5)
    o0 = env._obs()
    for p in env.humans.people.values():
        p.pose[:2] = p.pose[:2] + np.array([0.7, -0.4])
        p.state = 'standing'
    env.objects_truth = {k: (v[0] + 0.3, v[1] - 0.2, v[2]) for k, v in env.objects_truth.items()}
    np.testing.assert_array_equal(o0, env._obs())
    env.close()


def test_detector_names_do_not_depend_on_true_presence():
    env = HRIEnv(level=0)
    env.reset(seed=0)
    before = env.detect_names()
    for n in env.humans.people:
        env.humans.park(n)
    assert env.detect_names() == before
    env.close()


def _bias_localizer(env, offset_xy):
    real = env.localizer.update

    def biased(odom, scan):
        e = np.array(real(odom, scan), float)
        e[:2] += offset_xy
        return e
    env.localizer.update = biased
    env.est = env.est + np.array([*offset_xy, 0.0])


def test_straight_truth_and_belief_diverge():
    """Caso real de divergencia: la pose ESTIMADA está 0.5 m al costado de la real. La habilidad llega según la
    verdad (ok) pero el robot no lo sabe (belief); a la memoria solo llega la creencia."""
    env = LaundryEnv(level=0)
    env.reset(seed=0)
    yaw = float(env.est[2])
    goal = foot_center(env.truth())
    lateral = np.array([-np.sin(yaw), np.cos(yaw)]) * 0.5
    _bias_localizer(env, lateral)
    res = env.skill_straight(goal, yaw)
    assert res.ok is True and res.observed() is False
    env.close()


def test_laundry_at_basket_uses_belief():
    """'at_basket' (observación) se escribe con la creencia de la aproximación, no con su verificación real."""
    env = LaundryEnv(level=3)
    env.reset(seed=0)
    env.skill_navigate = lambda *a, **k: SkillResult(ok=True, belief=True)
    env.skill_straight = lambda *a, **k: SkillResult(ok=True, belief=False)
    env._approach_basket()
    assert env.mem['at_basket'] is False
    env.skill_straight = lambda *a, **k: SkillResult(ok=False, belief=True)
    env._approach_basket()
    assert env.mem['at_basket'] is True
    env.close()


def test_gpsr_believed_done_uses_belief():
    """'believed_done' (observación) de un comando 'go' sigue a la pose estimada, no a la llegada real."""
    from solverbot_rl.challenges.gpsr_grammar import Command
    env = GPSREnv(level=1)
    env.reset(seed=0)
    loc = next(k for k in env.arena.locations if k in ('kitchen', 'bedroom'))
    env.received = [Command(verb='go', location=loc)]
    env.cur = 0
    env.skill_navigate = lambda *a, **k: SkillResult(ok=True, belief=False)
    env._go_target(env.received[0], {})
    assert 0 not in env.mem['believed_done']
    env.skill_navigate = lambda *a, **k: SkillResult(ok=False, belief=True)
    env._go_target(env.received[0], {})
    assert 0 in env.mem['believed_done']
    env.close()


@pytest.mark.parametrize('cls', ENVS)
def test_nav_pose_source_estimate_by_default_and_validated(cls):
    env = cls(level=0)
    assert env.nav_pose_source == 'estimate'
    env.close()
    with pytest.raises(ValueError):
        cls(level=0, nav_pose_source='Estimate')
    with pytest.raises(ValueError):
        cls(level=0, nav_pose_source='')


def test_truth_navigation_is_explicit_and_reported():
    env = HRIEnv(level=0, nav_pose_source='truth')
    _, info = env.reset(seed=0)
    assert info['nav_pose_source'] == 'truth'
    np.testing.assert_array_equal(env._nav_pose(), env.truth())
    env.close()
    env = HRIEnv(level=0)
    env.reset(seed=0)
    assert env._nav_pose() is env.est
    env.close()


def test_door_align_is_never_skipped_while_turning():
    """Regresión de seguridad hallada al revalidar (HRI N5, semilla 1004): al llegar al punto previo de una puerta con el
    rumbo pasando por la tolerancia, el controlador saltaba a 'cross' en el mismo llamado (sin pausa ni robot quieto).
    Ahora entrar en 'align' devuelve una detención y la alineación se verifica en el llamado siguiente."""
    import math
    from solverbot_rl.challenges.nav import NavController, Waypoint
    pre = Waypoint(1.0, 2.0, 'door_pre', heading=math.pi / 2)
    post = Waypoint(1.0, 3.2, 'door_post', heading=math.pi / 2)
    nav = NavController([pre, post])
    from solverbot_rl.challenges.arena import AXLE
    yaw = math.pi / 2                               # justo sobre el eje: el salto viejo ocurría aquí
    base = np.array([1.0, 2.0]) - np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]) @ AXLE
    v, w = nav.act(np.array([*base, yaw]))
    assert (v, w) == (0.0, 0.0) and nav.mode == 'align' and nav.k == 0
    v, w = nav.act(np.array([*base, yaw]))
    assert nav.mode == 'cross' and nav.k == 1
