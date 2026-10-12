"""The paddle must not swing through the table.

A disc tilted t off horizontal hangs R*sin(t) below its own centre, so the
face centre clearing the table says nothing about the rim. Measured before
this was checked: half of all swings drove the blade under the surface, the
worst by 100 mm.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.pose import blade_lowest, clears_table, windup_pose  # noqa: E402

R = 0.055


def unit(v):
    v = np.array(v, float)
    return v / np.linalg.norm(v)


def test_a_flat_face_hangs_nothing_below_its_centre():
    assert blade_lowest([0, 0, 0.10], [0, 0, 1], R) == pytest.approx(0.10)


def test_a_vertical_face_hangs_a_full_radius_below():
    assert blade_lowest([0, 0, 0.10], [1, 0, 0], R) == pytest.approx(0.10 - R)


def test_a_tilted_face_hangs_r_sin_t_below():
    for deg in (30.0, 45.0, 60.0):
        n = unit([np.sin(np.radians(deg)), 0.0, np.cos(np.radians(deg))])
        expected = 0.12 - R * np.sin(np.radians(deg))
        assert blade_lowest([0, 0, 0.12], n, R) == pytest.approx(expected, abs=1e-9)


def test_the_strike_band_floor_is_not_safe_on_its_own():
    """30 mm is inside the strike band but a steep blade still fouls there."""
    steep = unit([1.0, 0.0, 0.4])
    assert not clears_table([0.3, 0.0, 0.030], steep, R)
    assert clears_table([0.3, 0.0, 0.030], [0, 0, 1], R)


def test_windup_draws_back_without_fouling_the_table():
    """The old wind-up dropped 45 mm in z, straight at the table."""
    strike = np.array([0.30, 0.0, 0.070])
    for deg in (20.0, 40.0, 60.0):
        n = unit([np.sin(np.radians(deg)), 0.0, np.cos(np.radians(deg))])
        naive = strike - np.array([0.0, 0.0, 0.045])
        drawn = windup_pose(strike, n, blade_radius=R)
        assert blade_lowest(drawn, n, R) >= blade_lowest(naive, n, R)


def test_windup_stays_near_the_strike_point():
    """It is a draw-back, not an escape: the arm has to swing from there."""
    strike = np.array([0.30, 0.0, 0.090])
    n = unit([0.5, 0.0, 0.86])
    drawn = windup_pose(strike, n, draw_back=0.055, blade_radius=R)
    assert np.linalg.norm(drawn - strike) == pytest.approx(0.055, abs=1e-6)
