"""The ball tracker's geometry, detector and filter, on synthetic frames."""

import cv2
import numpy as np
import pytest

from hw.cam_calibrate import fit
from hw.vision import BALL_R, GRAVITY, PADDLE_R, BallDetector, BallFilter, CameraModel, Tracker


def side_camera() -> CameraModel:
    """A camera 1 m off to the arm's side, level, looking at the workspace."""
    pos = np.array([0.25, -1.0, 0.30])
    fwd = np.array([0.0, 1.0, 0.0])
    down = np.array([0.0, 0.0, -1.0])
    right = np.cross(down, fwd)
    R = np.vstack([right, down, fwd])
    return CameraModel(f=900.0, cx=640, cy=360, R=R, t=-R @ pos)


def scene(rng, size=(720, 1280)) -> np.ndarray:
    """A cluttered background of coloured patches - wood, fabric, skin - as
    colourful as the real room measured around the arm (the table's chroma ~40)."""
    hsv = np.stack([rng.integers(0, 180, (size[0] // 40, size[1] // 40)),
                    rng.integers(120, 230, (size[0] // 40, size[1] // 40)),
                    rng.integers(60, 200, (size[0] // 40, size[1] // 40))], axis=-1).astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(img, (7, 7), 0)


def draw_ball(img, u, v, r, colour=(235, 235, 235)):
    out = img.copy()
    cv2.circle(out, (int(round(u)), int(round(v))), int(round(r)), colour, -1, cv2.LINE_AA)
    return out


def test_ball_position_round_trip():
    cam = side_camera()
    p = np.array([0.3, 0.05, 0.45])
    (uv,), _ = cam.project(p[None])
    back = cam.ball_in_world(uv[0], uv[1], cam.radius_px(p))
    assert np.allclose(back, p, atol=1e-3)


def test_depth_noise_dominates():
    cam = side_camera()
    C = cam.measurement_cov(np.array([0.3, 0.0, 0.3]))
    # The camera looks along +y, so y is depth.
    assert C[1, 1] > 100 * C[0, 0] and np.isclose(C[0, 0], C[2, 2])


def test_detector_finds_a_white_ball():
    rng = np.random.default_rng(0)
    bg = scene(rng)
    det = BallDetector()
    det(bg)  # learn the background
    d = det(draw_ball(bg, 500, 300, 15))
    assert d is not None
    assert abs(d.u - 500) < 1.5 and abs(d.v - 300) < 1.5
    assert d.r == pytest.approx(15, rel=0.12)


def test_detector_ignores_coloured_blobs():
    rng = np.random.default_rng(1)
    bg = scene(rng)
    det = BallDetector()
    det(bg)
    # A blue ball thrown in: moving, round, but not colourless. (Still grey
    # patches of the background can pass alone; a track never starts on them.)
    assert det(draw_ball(bg, 500, 300, 15, colour=(200, 60, 30)), min_moving=0.3) is None


def test_detector_finds_a_moving_grey_ball_against_a_bright_window():
    """Backlit: the ball is darker than what is behind it, so only motion finds it."""
    bg = np.full((720, 1280, 3), 245, np.uint8)
    det = BallDetector()
    det(bg)
    d = det(draw_ball(bg, 640, 360, 14, colour=(150, 150, 150)), min_moving=0.3)
    assert d is not None and abs(d.u - 640) < 2


def test_filter_follows_a_bounce():
    cam = side_camera()
    f = BallFilter()
    rng = np.random.default_rng(2)
    dt, p, v = 1 / 30, np.array([0.3, 0.0, 0.5]), np.array([0.0, 0.0, 0.0])
    for k in range(60):
        p, v = p + v * dt + 0.5 * GRAVITY * dt * dt, v + GRAVITY * dt
        if p[2] < 0.25 and v[2] < 0:  # the paddle sends it back up
            v[2] = -0.85 * v[2]
        z = rng.multivariate_normal(p, cam.measurement_cov(p))
        assert f.update(k * dt, z, cam.measurement_cov(p))
    assert f.bounces >= 1
    assert np.linalg.norm(f.pos - p) < 0.03
    assert abs(f.vel[2] - v[2]) < 0.4


def test_calibration_recovers_a_known_camera():
    true = side_camera()
    rng = np.random.default_rng(3)
    P = np.c_[rng.uniform(0.1, 0.45, 16), rng.uniform(-0.3, 0.3, 16), rng.uniform(0.15, 0.55, 16)]
    uv, _ = true.project(P)
    dist = np.linalg.norm(true.to_cam(P), axis=1)
    pts = [{"p": p.tolist(), "uv": (u + rng.normal(0, 1.0, 2)).tolist(), "major": 2 * PADDLE_R * true.f / d}
           for p, u, d in zip(P, uv, dist)]
    m, stats = fit(pts, 1280, 720)
    assert stats["rms_px"] < 2.5
    assert np.linalg.norm(m.position - true.position) < 0.03
    assert m.f == pytest.approx(true.f, rel=0.03)


def test_tracker_ignores_things_outside_the_workspace():
    cam = side_camera()
    tr = Tracker(cam=cam)
    rng = np.random.default_rng(4)
    bg = scene(rng)
    tr(bg, 0.0)
    # A white disc 3 m away, behind the arm: right size for nothing in the workspace.
    far = np.array([0.3, 2.0, 0.3])
    (uv,), _ = cam.project(far[None])
    for k in range(1, 6):
        r = tr(draw_ball(bg, uv[0] + 8 * k, uv[1], cam.radius_px(far)), k / 30)
    assert not r["live"]


def test_tracker_picks_up_a_ball_resting_in_the_workspace():
    cam = side_camera()
    tr = Tracker(cam=cam)
    bg = scene(np.random.default_rng(5))
    rest = np.array([0.3, 0.1, BALL_R])
    (uv,), _ = cam.project(rest[None])
    bg[int(uv[1]) - 10:, :] = (90, 140, 200)  # it sits on the wooden table
    frame = draw_ball(bg, uv[0], uv[1], cam.radius_px(rest))
    for k in range(5):
        r = tr(frame, k / 30)
    assert r["live"]
    err = r["pos"] - rest
    # The camera looks along +y: depth comes from the ball's size and is the noisy axis.
    assert abs(err[0]) < 0.01 and abs(err[2]) < 0.02 and abs(err[1]) < 0.08
    assert abs(r["vel"][2]) < 0.3  # held up by the table, not falling


def test_paddle_goes_under_a_held_ball_in_sim():
    from hw.ball_point import HOVER, face, plan_under
    from hw.so101 import SimCheck
    from sim.keepup import KeepUpEnv

    class SimArm:
        check = SimCheck()

        def q(self):
            return KeepUpEnv().q0

    arm = SimArm()
    ball = np.array([0.28, 0.08, 0.35])
    q, target, why = plan_under(arm, ball)
    assert q is not None, why
    fc, nrm = face(arm.check.world, q)
    assert np.hypot(*(fc - ball)[:2]) < 0.01
    assert abs(ball[2] - BALL_R - HOVER - fc[2]) < 0.01
    assert nrm[2] > np.cos(np.radians(6))


def test_refit_recovers_a_shifted_camera_and_radius_bias(tmp_path, monkeypatch):
    import json

    import hw.ball_point as bp
    import hw.cam_calibrate as cc

    true = side_camera()
    true.radius_scale = 1.08  # the detector reports radii 8% small
    rng = np.random.default_rng(7)
    T = np.c_[rng.uniform(0.15, 0.4, 10), rng.uniform(-0.2, 0.2, 10), rng.uniform(0.15, 0.4, 10)]
    uv, _ = true.project(T)
    pts = [{"uv": (u + rng.normal(0, 0.7, 2)).tolist(), "r": true.radius_px(p) + rng.normal(0, 0.3),
            "truth": p.tolist()} for u, p in zip(uv, T)]
    B = np.c_[rng.uniform(0.1, 0.4, 12), rng.uniform(-0.2, 0.2, 12), rng.uniform(0.1, 0.45, 12)]
    buv, _ = true.project(B)
    bd = np.linalg.norm(true.to_cam(B), axis=1)
    (tmp_path / "points.json").write_text(json.dumps({"points": [
        {"p": p.tolist(), "uv": u.tolist(), "major": 2 * 0.055 * true.f / d} for p, u, d in zip(B, buv, bd)]}))
    monkeypatch.setattr(cc, "OUT", tmp_path)
    # Start from a camera 3 cm off, with a 5% wrong focal length and no radius correction.
    start = CameraModel(f=true.f * 1.05, cx=true.cx, cy=true.cy, R=true.R, t=true.t + [0.03, -0.02, 0.0])
    m, stats = bp.refit(pts, start)
    # What is left is the measurement noise itself: 0.3 px on a 16 px ball is 2 cm of depth.
    assert stats["after_cm_rms"] < 0.5 * stats["before_cm_rms"]
    assert np.linalg.norm(m.position - true.position) < 0.015
    assert m.radius_scale == pytest.approx(1.08, rel=0.03)


def wood_table(rng, size=(720, 1280)) -> np.ndarray:
    """The real table's colour - the same hue as an orange ball, half as
    saturated (HSV 11, 136-158, 213) - with grain, and a grey wall above."""
    h, w = size
    hsv = np.empty((h, w, 3), np.uint8)
    hsv[..., 0] = 11
    hsv[..., 1] = np.clip(147 + rng.normal(0, 8, (h, w)), 0, 255)
    hsv[..., 2] = np.clip(205 + 15 * np.sin(np.arange(w) / 9.0)[None, :] + rng.normal(0, 4, (h, w)), 0, 255)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img[: h // 3] = (205, 210, 215)
    return cv2.GaussianBlur(img, (5, 5), 0)


def orange_ball(img, u, v, r):
    out = img.copy()
    cv2.circle(out, (int(round(u)), int(round(v))), int(round(r)), (20, 110, 250), -1, cv2.LINE_AA)
    return out


@pytest.mark.parametrize("light", ["neutral", "warm", "dark"])
def test_orange_ball_on_the_wooden_table(light):
    from hw.lighting_study import relight
    from hw.vision import DetectorConfig

    rng = np.random.default_rng(8)
    frame = orange_ball(wood_table(rng), 640, 500, 18)
    if light == "warm":
        frame = relight(frame, "warm", rng)
    elif light == "dark":
        frame = relight(frame, "under_4x", rng)
    det = BallDetector(DetectorConfig(ball="orange"))
    for _ in range(3):
        d = det(frame)
    assert d is not None and abs(d.u - 640) < 2 and abs(d.v - 500) < 2
    assert d.r == pytest.approx(18, rel=0.1)


def test_orange_detector_ignores_the_table_itself():
    from hw.vision import DetectorConfig

    det = BallDetector(DetectorConfig(ball="orange"))
    table = wood_table(np.random.default_rng(9))
    for _ in range(3):
        d = det(table)
    assert d is None


def test_off_axis_ball_round_trip_is_exact():
    """Off the optical axis a ball images as an ellipse, bigger and shifted
    outward; the camera model inverts that exactly."""
    cam = side_camera()
    for p in ([0.3, 0.05, 0.45], [0.75, 0.0, 0.70], [-0.2, 0.0, -0.05]):
        p = np.array(p)
        u, v, r = cam.sphere_image(cam.to_cam(p))
        assert np.allclose(cam.ball_in_world(u, v, r), p, atol=1e-6)
    # On the axis the outline is the familiar circle of radius f tan(asin(R/d)).
    d = 0.6
    u, v, r = cam.sphere_image([0.0, 0.0, d])
    assert (u, v) == pytest.approx((cam.cx, cam.cy)) and r == pytest.approx(cam.f * np.tan(np.arcsin(BALL_R / d)))
    # 30 degrees off-axis it is several percent bigger than that.
    off = np.array([d * np.tan(np.radians(30)), 0.0, d])
    assert cam.sphere_image(off)[2] > 1.04 * cam.f * np.tan(np.arcsin(BALL_R / np.linalg.norm(off)))
