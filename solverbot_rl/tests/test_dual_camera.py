"""Pruebas de la auditoría de cámaras y de los diagnósticos visión/control (sin simulador)."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'scripts'))
from solverbot_rl.camera_probe import HEAD_FRAME, cube_corners, look_at, project, visibility  # noqa: E402
from solverbot_rl.config import load_urdf  # noqa: E402
from solverbot_rl.envs.pick_place import SolverBotPickPlaceEnv  # noqa: E402
from solverbot_rl.imitation.experiment import env_kwargs, level_index, load_experiment  # noqa: E402
from solverbot_rl.imitation.rollout import OBS_SLICES, run_episode  # noqa: E402
from solverbot_rl.kinematics import UrdfChain  # noqa: E402
from solverbot_rl.perception import locate_object  # noqa: E402

RED = (0.85, 0.10, 0.10)
K = np.array([[116.4, 0, 80.0], [0, 116.4, 60.0], [0, 0, 1]])


def optical_pose(origin, forward):
    """cam_to_base de un frame óptico (z adelante, y abajo) en `origin` mirando a `forward` (horizontal o inclinado)."""
    z = np.asarray(forward, float) / np.linalg.norm(forward)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    t = np.eye(4)
    t[:3, :3] = np.stack([x, y, z], axis=1)
    t[:3, 3] = origin
    return t


def render(cam, truth, depth_value=None, bg_depth=3.0):
    """Imagen sintética 160x120: fondo gris y la cara del cubo (4 cm) en su proyección; profundidad plana."""
    rgb = np.full((120, 160, 3), 120, np.uint8)
    depth = np.full((120, 160), bg_depth, np.float32)
    c = project(cube_corners(truth, 0.0, (0.04,) * 3), cam, K)
    u0, u1 = int(c[:, 0].min()), int(np.ceil(c[:, 0].max()))
    v0, v1 = int(c[:, 1].min()), int(np.ceil(c[:, 1].max()))
    rgb[max(v0, 0):v1, max(u0, 0):u1] = (217, 25, 25)
    zc = project(np.asarray(truth, float)[None], cam, K)[0, 2]
    depth[max(v0, 0):v1, max(u0, 0):u1] = zc - 0.02 if depth_value is None else depth_value
    return rgb, depth


def test_projection_visibility_and_rgbd_roundtrip():
    cam = optical_pose([0, 0, 1.0], [1, 0, -0.3])
    truth = np.array([0.8, 0.05, 0.76])
    rgb, depth = render(cam, truth)
    est = locate_object(rgb, depth, K, cam, RED)
    v = visibility(rgb, depth, K, cam, truth, 0.0, (0.04,) * 3, RED, est)
    assert v['status'] == 'visible' and v['detected'] and v['mask_to_projection_px'] < 1.5
    # la RGB-D estima la cara visible: a lo sumo media arista de sesgo hacia la cámara
    assert np.linalg.norm(est - truth) < 0.025


def test_visibility_classifies_failures():
    cam = optical_pose([0, 0, 1.0], [1, 0, -0.3])
    truth = np.array([0.8, 0.05, 0.76])
    rgb, depth = render(cam, truth)
    blank = np.full_like(rgb, 120)
    # algo más cerca sobre la proyección (placa / brazo) y sin píxeles del cubo -> tapado
    near = np.full_like(depth, 0.3)
    assert visibility(blank, near, K, cam, truth, 0.0, (0.04,) * 3, RED)['status'] == 'occluded'
    neg = np.full_like(depth, -np.inf)                       # más cerca que el plano near del sensor
    assert visibility(blank, neg, K, cam, truth, 0.0, (0.04,) * 3, RED)['status'] == 'occluded'
    side = optical_pose([0, 0, 1.0], [0, 1, 0])               # el cubo queda a 90° del eje óptico
    assert visibility(blank, depth, K, side, truth, 0.0, (0.04,) * 3, RED)['status'] == 'out_of_fov'
    back = optical_pose([0, 0, 1.0], [-1, 0, 0])
    assert visibility(blank, depth, K, back, truth, 0.0, (0.04,) * 3, RED)['status'] == 'behind'


def test_look_at_points_head_optical_axis():
    ch = UrdfChain(load_urdf())
    js = {j: 0.0 for j in ch.joints}
    target = np.array([2.0, 1.2, 0.9])                        # dentro de los límites del pan/tilt
    pan, tilt = look_at(ch, js, target)
    cam = ch.link_pose(HEAD_FRAME, dict(js, pan_joint=pan, tilt_joint=tilt))
    d = (target - cam[:3, 3]) / np.linalg.norm(target - cam[:3, 3])
    assert np.degrees(np.arccos(np.clip(d @ cam[:3, 2], -1, 1))) < 3.0
    lo, hi = ch.limits(['pan_joint', 'tilt_joint'])
    pan, tilt = look_at(ch, js, [0.0, 0.3, 0.0])              # casi debajo: el tilt se recorta a su límite
    assert lo[1] <= tilt <= hi[1] and np.isclose(tilt, hi[1])


def test_branch_switch_detection():
    from analyze_branch import switches
    # acciones de joint4 de la semilla 902003 (pasos 0-19): sale de la saturación en 12, cruza 0 en 16
    a = np.array([-0.53, -0.73, -0.91, -0.97, -0.97, -0.98, -0.99, -0.99, -0.98, -0.96, -0.94, -0.91, -0.76, -0.57,
                  -0.43, -0.29, 0.06, 0.47, 0.85, 0.95])
    assert switches(a) == [{'leave': 12, 'zero_cross': 16, 'reach_other': 18, 'dir': '-1->+1'}]
    assert switches(np.full(30, 0.99)) == []
    two = np.r_[np.full(5, 1.0), np.full(5, -1.0), np.full(5, 1.0)]
    assert [s['dir'] for s in switches(two)] == ['+1->-1', '-1->+1']


@pytest.fixture(scope='module')
def kin_env():
    env = SolverBotPickPlaceEnv(backend='kinematic', **env_kwargs(load_experiment('reach_bc_v1')))
    yield env
    env.close()


def test_counterfactual_object_override(kin_env):
    from diag_obs_counterfactual import ObjectObsOverride
    obs, _ = kin_env.reset(seed=3, options={'level': level_index(kin_env, 'reach_var_m')})
    truth = kin_env.backend.object_pose_truth()
    tcp = obs[OBS_SLICES['tcp']]
    seen = obs.copy()
    seen[16] = 1.0                                            # paso con detección
    o = obs.copy()
    o[16] = 0.0                                               # paso sin detección
    o_before = o.copy()
    pr = ObjectObsOverride('perception', kin_env)
    assert np.array_equal(pr(o), o)
    orc = ObjectObsOverride('oracle', kin_env)
    out = orc(o)
    assert out[16] == 1.0 and np.allclose(out[OBS_SLICES['obj_rel']] + tcp, truth, atol=1e-6)
    hv = ObjectObsOverride('hold_valid', kin_env)
    assert hv(o)[16] == 0.0                                   # nunca hubo detección: no inventa validez
    assert hv(seen)[16] == 1.0 and hv(o)[16] == 1.0
    assert np.array_equal(hv(o)[OBS_SLICES['obj_rel']], o[OBS_SLICES['obj_rel']])
    fe = ObjectObsOverride('first_estimate', kin_env)
    first = seen[OBS_SLICES['obj_rel']] + tcp
    fe(seen)
    moved = seen.copy()
    moved[OBS_SLICES['obj_rel']] += 0.05
    assert np.allclose(fe(moved)[OBS_SLICES['obj_rel']] + tcp, first, atol=1e-6)
    # se modifican copias: la observación que entrega el entorno no cambia
    assert np.array_equal(o, o_before)


def test_on_step_hook_does_not_change_rollout(kin_env):
    lvl = level_index(kin_env, 'reach_var_m')
    rng = np.random.default_rng(0)
    acts = rng.uniform(-1, 1, (200, 7)).astype(np.float32)

    def policy():
        it = iter(acts)
        return lambda _obs: next(it)

    calls = []
    a1, s1 = run_episode(kin_env, 11, lvl, act=policy(), rng=np.random.default_rng(11))
    a2, s2 = run_episode(kin_env, 11, lvl, act=policy(), rng=np.random.default_rng(11),
                         on_step=lambda e, o, a: calls.append(a is None))
    for k in ('obs', 'actions', 'next_obs', 'dist', 'obj_valid'):
        assert np.array_equal(a1[k], a2[k], equal_nan=True), k
    assert calls[0] and not any(calls[1:]) and len(calls) == len(a1['actions']) + 1
    assert s1['success'] == s2['success'] and s1['length'] == s2['length']


def test_branch_guard_commits_and_limits_reversal():
    from solverbot_rl.imitation.branch_guard import BranchCommitGuard
    lo, hi = -np.ones(6) * 2, np.ones(6) * 2
    g = BranchCommitGuard(lo, hi, commit=0.15, max_reverse=0.2)

    def obs(q4):
        o = np.zeros(31, np.float32)
        o[3] = 2 * (q4 - lo[3]) / (hi[3] - lo[3]) - 1
        return o
    a = np.zeros(7, np.float32)
    a[3] = 1.0
    assert np.array_equal(g(obs(-0.6), a), a)               # sin compromiso todavía: no interviene
    back = a.copy()
    back[3] = -1.0
    assert g(obs(-0.55), back)[3] == -1.0                   # |Δq4| = 0.05 < commit
    g(obs(-0.40), a)                                        # Δq4 = +0.20 -> rama +
    out = g(obs(-0.35), back)
    assert g.summary()['branch'] == 1 and np.isclose(out[3], -0.2) and g.summary()['interventions'] == 1
    assert np.array_equal(out[[0, 1, 2, 4, 5, 6]], back[[0, 1, 2, 4, 5, 6]])   # solo toca joint4
    g.reset()
    assert g.summary() == {'branch': 0, 'commit_step': None, 'interventions': 0}
