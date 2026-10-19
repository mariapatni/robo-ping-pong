"""The hand-eye fit recovers a known camera, joint offsets and paddle mount
in the sim (render-free: exact corners plus 0.15 px noise; the rendered proof
is `python -m hw.handeye --sim`)."""

import pytest

pytest.importorskip("scipy")


def test_hand_eye_recovers_the_truth():
    from hw.handeye import run_sim
    r = run_sim(seed=0, n_fit=40, n_held=12, render=False, out=None, log=lambda *_: None, exact_lens=True)
    assert r["camera_position_err_mm"] < 1.0 and r["camera_angle_err_deg"] < 0.05
    assert max(abs(e) for e in r["joint_offset_err_deg"].values()) < 0.1
    assert r["mount_shift_err_mm"] < 0.3 and r["mount_tilt_err_deg"] < 0.1
    assert r["reproj_rms_px"] < 0.3 and r["face_max_mm"] < 0.5
    assert r["passed"]
