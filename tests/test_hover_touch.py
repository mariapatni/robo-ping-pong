"""The hover-the-ball touch test in the sim: the paddle comes up flat under
where the camera says the ball is, and touches its bottom within a centimetre."""

import pytest

pytest.importorskip("stag")


def test_paddle_touches_the_bottom_of_the_ball():
    from hw.hover_touch import run_sim
    trials, s = run_sim(n=3, seed=0, out=None, log=lambda *_: None)
    assert s["touched"] == s["spots"] == 3
    assert s["passed"] == 3
    assert s["sideways_max_mm"] < 5 and s["height_max_mm"] < 3
    # A flat face meets a sphere at its bottom: where it touched is the truth.
    assert s["contact_err_max_mm"] < 0.5 and s["tilt_max_deg"] < 1
    assert 0.98 < s["radius_scale"] < 1.03
