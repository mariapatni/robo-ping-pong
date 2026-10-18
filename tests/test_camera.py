"""The camera interface and lens calibration, with no camera attached.

Calibration is tested the way it can fail: views of a board seen through a
known lens must solve back to that lens, and the printable board must be one
the corner finder actually finds.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from hw.calibrate_camera import board_points, find_corners, make_board, solve  # noqa: E402
from hw.camera import Camera, CameraSpec  # noqa: E402

K_TRUE = np.array([[910.0, 0, 652.0], [0, 905.0, 355.0], [0, 0, 1]])
DIST_TRUE = np.array([0.08, -0.15, 0.0, 0.0, 0.0])
SIZE = (1280, 720)


def synthetic_views(n=20, seed=0):
    """Board corners as the known lens would see them, from varied poses."""
    rng = np.random.default_rng(seed)
    obj = board_points(9, 6, 25.0)
    centre = obj.mean(axis=0)
    views = []
    while len(views) < n:
        rvec = rng.uniform(-0.5, 0.5, 3)
        tvec = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.08, 0.08), rng.uniform(0.35, 0.8)])
        R, _ = cv2.Rodrigues(rvec)
        t = tvec - R @ centre
        img, _ = cv2.projectPoints(obj, cv2.Rodrigues(R)[0], t, K_TRUE, DIST_TRUE)
        pts = img.reshape(-1, 2)
        if (pts[:, 0] > 5).all() and (pts[:, 0] < SIZE[0] - 5).all() and (pts[:, 1] > 5).all() and (pts[:, 1] < SIZE[1] - 5).all():
            views.append((img + rng.normal(0, 0.1, img.shape)).astype(np.float32))
    return views


def test_solve_recovers_a_known_lens():
    cal = solve(synthetic_views(), SIZE, 9, 6, 25.0)
    K = np.array(cal["K"])
    assert K[0, 0] == pytest.approx(K_TRUE[0, 0], rel=0.01)
    assert K[1, 1] == pytest.approx(K_TRUE[1, 1], rel=0.01)
    assert K[0, 2] == pytest.approx(K_TRUE[0, 2], abs=5)
    assert K[1, 2] == pytest.approx(K_TRUE[1, 2], abs=5)
    assert cal["dist"][0] == pytest.approx(DIST_TRUE[0], abs=0.03)
    assert cal["rms_px"] < 0.3
    assert cal["hfov_deg"] == pytest.approx(np.degrees(2 * np.arctan(640 / 910)), abs=0.5)


def test_the_printable_board_is_found_by_the_corner_finder(tmp_path):
    img = cv2.imread(str(make_board(tmp_path / "board.png", 9, 6, 25.0, dpi=100)), cv2.IMREAD_GRAYSCALE)
    corners = find_corners(img, 9, 6)
    assert corners is not None and len(corners) == 54
    # Neighbouring corners one square apart: 25 mm at 100 dpi is ~98 px.
    step = np.linalg.norm(corners[1, 0] - corners[0, 0])
    assert step == pytest.approx(25 / 25.4 * 100, rel=0.05)


def test_a_spec_round_trips_and_refuses_intrinsics_until_calibrated(tmp_path):
    spec = CameraSpec("test", "avfoundation", "Some Camera", 1280, 720, 30.0, notes="x")
    path = spec.save(tmp_path / "test.json")
    back = CameraSpec.load(path)
    assert back == spec
    with pytest.raises(ValueError, match="not calibrated"):
        back.intrinsics()
    back.calibration = {"K": K_TRUE.tolist(), "dist": DIST_TRUE.tolist()}
    K, dist = back.intrinsics()
    assert np.allclose(K, K_TRUE)


def test_the_file_backend_plays_a_recording_then_ends(tmp_path):
    imageio = pytest.importorskip("imageio.v2")
    path = tmp_path / "clip.mp4"
    imageio.mimsave(path, [np.full((64, 96, 3), 10 * i, np.uint8) for i in range(8)], fps=30, macro_block_size=1)
    with Camera(CameraSpec("clip", "file", str(path))) as cam:
        frames = []
        while True:
            f, t = cam.wait_new(timeout=2.0)
            if f is None:
                break
            frames.append((f, t))
    assert 6 <= len(frames) <= 8
    assert frames[0][0].shape == (64, 96, 3)
    assert all(b[1] > a[1] for a, b in zip(frames, frames[1:]))  # times move forward
