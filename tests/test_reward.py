"""The cost function: underhand pays, overhand does not, rim hits score nothing.

The tests that matter here are the ORDERING tests. Individual weights are
arguable; the order they put outcomes in is the actual design, and it is what
would silently invert if someone retuned a weight.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.config import TableConfig  # noqa: E402
from sim.reward import (  # noqa: E402
    StrokeWeights,
    exchange_reward,
    face_hit_quality,
    lift,
    stroke_term,
)

TABLE = TableConfig()
AIM = (0.95, 0.0)


def unit(v):
    v = np.array(v, float)
    return v / np.linalg.norm(v)


FACE_UP = unit([0.3, 0.0, 0.95])
FACE_DOWN = unit([0.3, 0.0, -0.95])
RISING = np.array([1.4, 0.0, 0.6])
FALLING = np.array([1.6, 0.0, -0.3])


def score(face_n, vel, landing, contact_n=None):
    return exchange_reward(
        contact_normal=face_n if contact_n is None else contact_n,
        face_normal=face_n, ball_vel_out=vel, landing_xy=landing,
        table=TABLE, aim_xy=AIM,
    )


def test_face_hit_quality_rejects_the_rim():
    assert face_hit_quality([0, 0, 1], [0, 0, 1]) == pytest.approx(1.0)
    assert face_hit_quality([0, 1, 0], [0, 0, 1]) == 0.0


def test_either_face_counts():
    """The blade is symmetric; returning with the back is legal."""
    assert face_hit_quality([0, 0, -1], [0, 0, 1]) == pytest.approx(1.0)


def test_lift_is_signed_through_horizontal():
    assert lift([0, 0, 1]) == pytest.approx(1.0)
    assert lift([0, 0, -1]) == pytest.approx(-1.0)
    assert lift([1, 0, 0]) == pytest.approx(0.0, abs=1e-9)


def test_face_up_but_chopping_down_is_not_underhand():
    """The gate is on the consequence: holding the face open while driving the
    ball downward must not score as an underhand lift."""
    w = StrokeWeights()
    assert stroke_term(FACE_UP, RISING, w) > 0.0
    assert stroke_term(FACE_UP, FALLING, w) == 0.0


def test_the_outcome_ordering_is_the_design():
    under_on = score(FACE_UP, RISING, AIM)["total"]
    under_off = score(FACE_UP, RISING, (1.05, 0.30))["total"]
    over_in = score(FACE_DOWN, FALLING, AIM)["total"]
    under_out = score(FACE_UP, RISING, (1.35, 0.0))["total"]
    rim = score(FACE_UP, RISING, AIM, contact_n=[0, 1, 0])["total"]
    miss = exchange_reward(table=TABLE)["total"]

    assert under_on > under_off > over_in > under_out > rim == miss == 0.0


def test_underhand_is_far_better_than_overhand():
    under = score(FACE_UP, RISING, AIM)["total"]
    over = score(FACE_DOWN, FALLING, AIM)["total"]
    assert under > 5 * over, f"underhand {under:.2f} vs overhand {over:.2f}"


def test_a_scored_overhand_still_beats_standing_still():
    """If this inverts, the cheapest way to dodge the overhand penalty is to
    never swing, and the policy learns to do nothing."""
    assert score(FACE_DOWN, FALLING, AIM)["total"] > exchange_reward(table=TABLE)["total"]


def test_height_is_not_rewarded():
    """A loop and a drive that land in the same place score the same."""
    loop = score(FACE_UP, np.array([1.0, 0.0, 2.0]), AIM)
    drive = score(FACE_UP, np.array([1.8, 0.0, 0.3]), AIM)
    assert loop["landing"] == pytest.approx(drive["landing"])


def test_landing_short_of_the_net_is_out():
    assert not score(FACE_UP, RISING, (0.30, 0.0))["in_bounds"]
    assert score(FACE_UP, RISING, (0.95, 0.0))["in_bounds"]


def test_landing_wide_is_out():
    assert not score(FACE_UP, RISING, (0.95, TABLE.half_width + 0.05))["in_bounds"]


def test_faults_are_never_negative():
    """Reward hack 9: a negative fault makes deliberate fouling an improvement."""
    for r in (exchange_reward(table=TABLE),
              score(FACE_UP, RISING, AIM, contact_n=[0, 1, 0]),
              score(FACE_DOWN, FALLING, (1.35, 0.0))):
        assert r["total"] >= 0.0
