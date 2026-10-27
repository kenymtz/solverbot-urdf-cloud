"""Pruebas de la recompensa y de los criterios de éxito, incluidas pruebas contra reward hacking.
Puras (sin simulador): construyen TaskState sintéticos."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solverbot_rl import task as T  # noqa: E402
from solverbot_rl.config import load_config  # noqa: E402

CFG = load_config()
TABLE_Z = CFG['task']['table_height']
HALF = CFG['world']['object']['size'][2] / 2
OBJ = np.array([0.22, 0.56, TABLE_Z + HALF])
DOWN = np.array([0.0, 0.0, -1.0])


def params(goal):
    return T.TaskParams.from_config(CFG, goal)


def state(tcp, obj=OBJ, g=0.035, closed=False, z=DOWN, vel=0.0, pts=None, limit=0):
    tcp = np.asarray(tcp, float)
    return T.TaskState(tcp_pos=tcp, tcp_z=np.asarray(z, float), obj_pos=np.asarray(obj, float), gripper_q=g,
                       gripper_cmd_closed=closed, obj_vel=vel, joints_at_limit=limit,
                       arm_points=np.asarray(pts if pts is not None else [tcp + [0, 0, 0.1]], float))


def run(states, goal, actions=None):
    p, f = params(goal), T.EpisodeFlags()
    f.best_phi = T.potential(states[0], p)
    total, out = 0.0, []
    for k in range(1, len(states)):
        a = np.zeros(7) if actions is None else actions[k]
        pa = np.zeros(7) if actions is None else actions[k - 1]
        r, terms, term, stalled = T.step_reward(states[k - 1], states[k], a, pa, p, f)
        total += r
        out.append((r, terms, term, stalled))
    return total, out, f


def test_shaping_cycle_does_not_pay():
    """Ir y volver repetidamente cerca del objeto no acumula recompensa positiva (shaping por potencial)."""
    a, b = OBJ + [0.10, 0, 0.10], OBJ + [0.02, 0, 0.05]
    cycle = [state(a), state(b)] * 20 + [state(a)]
    total, out, _ = run(cycle, 'grasp')
    shaping = sum(o[1]['shaping'] for o in out)
    assert shaping <= 1e-6
    assert total < 0


def test_idle_is_negative():
    total, out, _ = run([state(OBJ + [0.1, 0, 0.1])] * 30, 'lift')
    assert total < 0
    assert not any(o[1]['success'] for o in out)


def test_empty_close_is_not_a_grasp():
    """Cerrar la pinza en el aire (dedos completamente cerrados) no cuenta como agarre."""
    s = [state(OBJ + [0, 0, 0.05]), state(OBJ + [0, 0, 0.05], g=0.0, closed=True)]
    _, out, f = run(s, 'grasp')
    assert not f.grasped_once and 'bonus_grasp' not in out[-1][1]


def test_grasp_requires_object_between_fingers():
    half_w = CFG['world']['object']['size'][0] / 2
    far = state(OBJ + [0.06, 0, 0], g=half_w, closed=True)        # dedos "bloqueados" pero el objeto no está ahí
    assert not T.fingers_blocked_by_object(far, params('grasp'))
    near = state(OBJ, g=half_w, closed=True)
    assert T.fingers_blocked_by_object(near, params('grasp'))


def test_grasp_bonus_only_once():
    half_w = CFG['world']['object']['size'][0] / 2
    seq = []
    for _ in range(5):
        seq += [state(OBJ, g=half_w, closed=True), state(OBJ, g=0.035, closed=False)]
    _, out, _ = run([state(OBJ + [0, 0, 0.05])] + seq, 'lift')
    assert sum('bonus_grasp' in o[1] for o in out) == 1


def test_pushing_into_target_is_not_success():
    """Empujar el objeto (sin levantarlo) hasta la zona objetivo NO es éxito."""
    p = params('place')
    tgt = np.array([*p.target_xy, OBJ[2]])
    path = [OBJ + (tgt - OBJ) * k / 10 for k in range(11)]
    states = [state(o + [-0.03, 0, 0], obj=o) for o in path] + [state(tgt + [0, 0, 0.12], obj=tgt)]
    _, out, f = run(states, 'place')
    assert not f.lifted_once
    assert not any(o[1]['success'] for o in out)


def test_place_success_requires_lift_release_and_rest():
    p = params('place')
    half_w = CFG['world']['object']['size'][0] / 2
    tgt = np.array([*p.target_xy, OBJ[2]])
    up = OBJ + [0, 0, 0.08]
    seq = [state(OBJ + [0, 0, 0.05]), state(OBJ, g=half_w, closed=True)]
    seq += [state(up, obj=up, g=half_w, closed=True)] * 2
    seq += [state(tgt + [0, 0, 0.08], obj=tgt + [0, 0, 0.08], g=half_w, closed=True)]
    seq += [state(tgt, obj=tgt, g=half_w, closed=True)]
    # soltar pero sin alejarse todavía: aún no es éxito
    seq += [state(tgt, obj=tgt, g=0.035, closed=False)]
    seq += [state(tgt + [0, 0, 0.08], obj=tgt, g=0.035, closed=False)]
    _, out, f = run(seq, 'place')
    assert f.lifted_once
    succ = [o[1]['success'] for o in out]
    assert succ[-1] == 1.0 and sum(succ) == 1.0
    assert out[-1][2]                                              # termina el episodio


def test_place_rejected_while_object_moving():
    p = params('place')
    tgt = np.array([*p.target_xy, OBJ[2]])
    f = T.EpisodeFlags(lifted_once=True)
    assert not T.placed_in_zone(state(tgt + [0, 0, 0.1], obj=tgt, vel=0.3), p, f)
    assert T.placed_in_zone(state(tgt + [0, 0, 0.1], obj=tgt, vel=0.0), p, f)


def test_drop_terminates_with_penalty():
    fallen = OBJ - [0, 0, 0.5]
    _, out, _ = run([state(OBJ + [0, 0, 0.05]), state(OBJ + [0, 0, 0.05], obj=fallen)], 'lift')
    r, terms, term, _ = out[-1]
    assert term and 'pen_drop' in terms and r < 0


def test_table_collision_terminates():
    under = OBJ + [0.05, 0.0, -0.05]                                # punto del brazo bajo la tapa de la mesa
    _, out, _ = run([state(OBJ + [0, 0, 0.1]), state(OBJ + [0, 0, 0.1], pts=[under])], 'grasp')
    assert out[-1][2] and 'pen_collision' in out[-1][1]


def test_action_rate_and_limits_penalised():
    s = [state(OBJ + [0.1, 0, 0.1])] * 3
    acts = [np.zeros(7), np.ones(7), -np.ones(7)]
    _, out, _ = run(s, 'reach', acts)
    assert out[-1][1]['pen_action_rate'] < out[0][1]['pen_action_rate'] <= 0
    _, out2, _ = run([s[0], state(OBJ + [0.1, 0, 0.1], limit=2)], 'reach')
    assert out2[-1][1]['pen_limit'] < 0


def test_no_progress_truncates():
    p = params('reach')
    n = p.no_progress_steps
    _, out, _ = run([state(OBJ + [0.2, 0, 0.2])] * (n + 2), 'reach')
    assert any(o[3] for o in out)
    assert any('pen_no_progress' in o[1] for o in out)


def test_hover_with_object_does_not_accumulate():
    """Mantener el objeto levantado indefinidamente no paga por paso (el bono de levantamiento es único)."""
    half_w = CFG['world']['object']['size'][0] / 2
    up = OBJ + [0, 0, 0.03]                    # levantado pero por debajo de lift_height: sin éxito
    seq = [state(OBJ + [0, 0, 0.05]), state(OBJ, g=half_w, closed=True)]
    seq += [state(up, obj=up, g=half_w, closed=True)] * 30
    _, out, _ = run(seq, 'lift')
    tail = sum(o[0] for o in out[5:])
    assert tail < 0


@pytest.mark.parametrize('goal', T.GOALS)
def test_potential_is_finite(goal):
    assert np.isfinite(T.potential(state(OBJ + [0.1, 0.1, 0.1]), params(goal)))
