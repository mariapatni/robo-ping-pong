"""The printed ChArUco board must read correctly from the face that was on the bed."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cad"))
import charuco  # noqa: E402


def detect(img):
    b, _ = charuco.board()
    det = cv2.aruco.CharucoDetector(b)
    corners, ids, mcorners, mids = det.detectBoard(img)
    return corners, ids, mids


def test_the_printed_face_reads_as_the_board():
    corners, ids, mids = detect(charuco.seen_from_printed_face())
    assert mids is not None and sorted(mids.ravel()) == list(range(16))
    assert ids is not None and len(ids) == (charuco.COLS - 1) * (charuco.ROWS - 1)


def test_the_unflipped_model_does_not():
    """If the mirroring were dropped, the markers would be unreadable mirror images."""
    img = charuco.raster(8.0, [(charuco.WIDTH - x1, charuco.HEIGHT - y1, charuco.WIDTH - x0, charuco.HEIGHT - y0)
                               for x0, y0, x1, y1 in charuco.model_rects()][::1])
    flipped = img[:, ::-1]  # the top of the model, seen from above
    _, _, mids = detect(flipped)
    assert mids is None or len(mids) == 0


def test_corners_land_on_the_grid():
    px = 8.0
    corners, ids, _ = detect(charuco.seen_from_printed_face(px))
    b, _ = charuco.board()
    truth = b.getChessboardCorners()[ids.ravel(), :2] + charuco.FRAME
    err = np.linalg.norm(corners.reshape(-1, 2) / px - truth, axis=1)
    assert err.max() < 0.15  # mm


def test_it_fits_half_the_bed():
    assert charuco.WIDTH <= 250 and charuco.HEIGHT <= 128
