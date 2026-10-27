"""Regresión de la orientación en la IK local del experto."""
from types import SimpleNamespace

import numpy as np
import pytest

import solverbot_rl.expert as expert_mod


@pytest.mark.parametrize("goal", ["reach", "grasp"])
def test_local_ik_uses_correct_orientation(goal, monkeypatch):
    obj = np.array([0.22, 0.56, 0.76])
    tcp_z = np.array([0.67, 0.0, -0.74])

    state = SimpleNamespace(
        tcp_pos=obj + np.array([0.02, 0.0, 0.08]),
        tcp_z=tcp_z,
    )

    params = SimpleNamespace(
        goal=goal,
        pregrasp_height=0.08,
        target_xy=np.array([0.30, 0.60]),
        table_height=0.74,
        obj_half_height=0.02,
    )

    inner_env = SimpleNamespace(
        _state=state,
        params=params,
        backend=SimpleNamespace(object_pose_truth=lambda: obj.copy()),
        chain=object(),
        arm_joints=[f"joint{i}" for i in range(1, 7)],
        _q_cmd=np.zeros(6),
        delta=0.05,
        cfg={"robot": {"lift": 0.0}},
    )

    captured = {}

    def fake_ik(*args, **kwargs):
        captured["axis"] = np.asarray(kwargs["approach_axis"])
        return np.zeros(6), 0.0

    monkeypatch.setattr(expert_mod, "solve_ik", fake_ik)

    expert = expert_mod.ScriptedExpert(
        SimpleNamespace(unwrapped=inner_env)
    )
    expert.act()

    expected = (
        expert_mod.APPROACH
        if goal == "reach"
        else tcp_z
    )

    np.testing.assert_allclose(captured["axis"], expected)
