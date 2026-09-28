"""The arm reaches what it is told to reach, and respects its limits.

This is deliberately the same shape as the open-loop replay test we will run
against the real hardware later: command a trajectory, compare the joint
angles that came back.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.arm import ARM_JOINTS, HOME, Arm  # noqa: E402

TOL_DEG = 0.5


@pytest.fixture(scope="module")
def arm():
    return Arm()


def test_model_shape(arm):
    assert len(ARM_JOINTS) == 5
    assert arm.qpos.shape == (5,)
    assert np.all(arm.lower < arm.upper)


def test_home_is_reachable(arm):
    arm.reset()
    assert np.allclose(arm.qpos, arm.clip(HOME), atol=1e-6)


@pytest.mark.parametrize(
    "target_deg",
    [
        (0, -35, 50, 30, 0),
        (45, -55, 70, 25, 40),
        (-45, -55, 70, 25, -40),
        (0, -80, 90, 40, 0),
    ],
)
def test_reaches_commanded_joint_targets(arm, target_deg):
    arm.reset()
    target = arm.clip(np.radians(target_deg))
    info = arm.goto(target, timeout=6.0)
    assert info["settled"], f"did not settle, max error {info['max_error_deg']:.3f} deg"
    assert info["max_error_deg"] < TOL_DEG


def test_targets_are_clipped_to_limits(arm):
    arm.reset()
    absurd = np.array([10.0, -10.0, 10.0, -10.0, 10.0])
    applied = arm.command(absurd)
    assert np.all(applied >= arm.lower - 1e-9)
    assert np.all(applied <= arm.upper + 1e-9)


def test_tip_moves_when_joints_move(arm):
    arm.reset()
    tip_home = arm.tip_pos()
    arm.goto(arm.clip(np.radians((60, -70, 80, 30, 0))), timeout=6.0)
    assert np.linalg.norm(arm.tip_pos() - tip_home) > 0.05
