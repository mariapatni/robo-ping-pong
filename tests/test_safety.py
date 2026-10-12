"""The table is a hard constraint, not a price.

A reward term can be paid: the optimiser weighs it against everything else and
violates when the payoff is larger. arXiv 2505.22974 measured that on a current
limit and found soft penalties were violated where a hard constraint was not.
On the real arm a table strike is not a bad outcome in the game, it is a broken
paddle, so it is enforced between the policy and the servos.
"""

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.config import EnvConfig  # noqa: E402
from sim.env import PingPongEnv  # noqa: E402
from sim.paddle import PaddleConfig  # noqa: E402
from sim.safety import SafetyLimits, TableGuard  # noqa: E402


@pytest.fixture(scope="module")
def world():
    w = PingPongEnv(cfg=EnvConfig(paddle=PaddleConfig()))
    w.tip_site = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_SITE, "paddle_face")
    return w


@pytest.fixture(scope="module")
def guard(world):
    return TableGuard(world)


def test_the_guard_watches_the_whole_arm_not_just_the_paddle(world, guard):
    """The forearm touched the table more often than the blade did, so a
    blade-only check misses most of the problem."""
    names = {mujoco.mj_id2name(world.model, mujoco.mjtObj.mjOBJ_BODY,
                               world.model.geom_bodyid[g]) for g in guard.arm_geoms}
    assert "lower_arm" in names
    assert "paddle" in names
    assert len(guard.arm_geoms) > 5


def test_a_pose_clear_of_the_table_is_allowed(world, guard):
    up = np.array([0.0, -1.2, 1.2, 0.0, 0.0])
    assert guard.is_safe(world.clip(up))


def test_the_guard_never_returns_an_unsafe_command(world, guard):
    """Over many random targets, whatever comes out must be safe - that is the
    whole contract."""
    rng = np.random.default_rng(0)
    world.reset(np.array([0.0, -1.2, 1.2, 0.0, 0.0]))
    unsafe_seen = 0
    for _ in range(40):
        target = world.clip(rng.uniform(world.lower, world.upper))
        if not guard.is_safe(target):
            unsafe_seen += 1
        out = guard.clamp(target, world.qpos)
        assert guard.is_safe(out), "the guard let an unsafe command through"
    assert unsafe_seen > 0, "the sample never proposed anything unsafe"


def test_clamping_moves_toward_the_target_not_away(world, guard):
    rng = np.random.default_rng(1)
    world.reset(np.array([0.0, -1.2, 1.2, 0.0, 0.0]))
    for _ in range(20):
        target = world.clip(rng.uniform(world.lower, world.upper))
        out = guard.clamp(target, world.qpos)
        along = (out - world.qpos) @ (target - world.qpos)
        assert along >= -1e-9, "the clamp pushed away from the requested motion"


def test_a_tighter_margin_is_never_less_safe(world):
    loose = TableGuard(world, SafetyLimits(table_margin=0.002))
    tight = TableGuard(world, SafetyLimits(table_margin=0.030))
    rng = np.random.default_rng(2)
    for _ in range(25):
        q = world.clip(rng.uniform(world.lower, world.upper))
        if tight.is_safe(q):
            assert loose.is_safe(q)
