"""Where the arm may hit, and feeds that cover all of it.

The two things worth pinning here are coverage and realism: feeds must reach
the whole underhand zone, not just the middle, and they must arrive at rally
speed rather than at the speed a ball dropped from height happens to have.
"""

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import replace  # noqa: E402

from sim.config import EnvConfig, TableConfig  # noqa: E402
from sim.env import PingPongEnv  # noqa: E402
from sim.feed import aim_feed, sample_targets  # noqa: E402
from sim.paddle import PaddleConfig  # noqa: E402
from sim.zones import grid, mirror, sweep_face, zones  # noqa: E402

TABLE = TableConfig()


@pytest.fixture(scope="module")
def world():
    w = PingPongEnv(cfg=EnvConfig(paddle=PaddleConfig()))
    w.tip_site = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_SITE, "paddle_face")
    return w


@pytest.fixture(scope="module")
def swept(world):
    return sweep_face(world, samples=(21, 11, 11, 9, 9))


def test_the_underhand_zone_is_a_subset_of_reach(world, swept):
    z = zones(world, TABLE, swept=swept)
    assert (z["underhand"] & ~z["reach"]).sum() == 0
    assert z["underhand"].sum() < z["reach"].sum()


def test_moving_the_arm_forward_covers_more_of_its_half(world, swept):
    half = int(round(TABLE.half_length / 0.02))
    back = zones(world, replace(TABLE, arm_inset=0.0), swept=swept)["underhand"][:half].sum()
    fwd = zones(world, replace(TABLE, arm_inset=0.15), swept=swept)["underhand"][:half].sum()
    assert fwd > back


def test_the_opponent_zone_is_a_rotation_not_a_reflection(world, swept):
    """A reflection would invert the joint chain's handedness; the arms are
    related by a 180 degree turn about the table centre."""
    z = zones(world, TABLE, swept=swept)
    assert np.array_equal(mirror(z["underhand"]), z["b_underhand"])
    assert np.array_equal(mirror(mirror(z["underhand"])), z["underhand"])


def test_the_target_zone_is_past_the_net(world, swept):
    z = zones(world, TABLE, swept=swept)
    half = int(round(TABLE.half_length / 0.02))
    assert z["target"][:half].sum() == 0
    assert z["target"].sum() > 0


def test_targets_are_sampled_across_the_whole_zone(world, swept):
    z = zones(world, TABLE, swept=swept)
    xe, ye = grid(TABLE)
    t = sample_targets(z["underhand"], xe, ye, 400, seed=1)
    # Not bunched down the middle: both wings and the deep band get sampled.
    assert (t[:, 1] < -0.20).any() and (t[:, 1] > 0.20).any()
    assert np.ptp(t[:, 0]) > 0.35  # ndarray.ptp() was removed in NumPy 2


def test_aimed_feeds_land_on_their_target_at_rally_speed(world, swept):
    z = zones(world, TABLE, swept=swept)
    xe, ye = grid(TABLE)
    targets = sample_targets(z["underhand"], xe, ye, 12, seed=7)
    feeds = [aim_feed(t, EnvConfig().ball, spawn_x=TABLE.x_far - 0.10,
                      table_e=TABLE.restitution, seed=i) for i, t in enumerate(targets)]
    miss = np.array([f.miss for f in feeds])
    arrival = np.array([f.arrival_speed for f in feeds])
    assert (miss < 0.020).mean() >= 0.9, f"aim error {miss*1000}"
    # A direct, un-bounced feed arrives at ~2.9 m/s on gravity alone. Staying
    # under 2.5 is evidence the solver is finding the bounced solution.
    assert arrival.max() < 2.5, f"arrival speeds {arrival}"
    assert arrival.min() > 0.8
