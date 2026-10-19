"""Each face of the tagged paddle must read as its own STag id, seen from outside."""

import sys
from pathlib import Path

import numpy as np
import pytest

stag = pytest.importorskip("stag")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cad"))
import paddle_tags  # noqa: E402


@pytest.fixture(scope="module")
def lay():
    return paddle_tags.layout()


@pytest.mark.parametrize("face", ["front", "back"])
def test_each_face_reads_its_id_from_outside(lay, face):
    img = paddle_tags.face_image(face, lay)
    corners, ids, _ = stag.detectMarkers(np.ascontiguousarray(img), paddle_tags.TAG_HD)
    assert ids is not None and list(np.ravel(ids)) == [lay["faces"][face]["id"]]


def test_the_front_must_be_mirrored(lay):
    """Laid out like the back - not mirrored - the hitting face, seen from
    outside, is a mirror image, and does not read as id 0."""
    unmirrored = dict(lay, faces=dict(lay["faces"], front=dict(lay["faces"]["front"], right=[0, -1, 0])))
    img = paddle_tags.face_image("front", unmirrored)
    _, ids, _ = stag.detectMarkers(np.ascontiguousarray(img), paddle_tags.TAG_HD)
    assert ids is None or lay["faces"]["front"]["id"] not in np.ravel(ids)


def test_markers_fit_the_blade_with_margin(lay):
    for f in lay["faces"].values():
        assert f["edge"] / np.sqrt(2) <= lay["white_r"] - paddle_tags.TAG_CORNER_CLEAR + 1e-9
    back = lay["faces"]["back"]["edge"]
    to_rib = lay["centre_xy"][0] - max(paddle_tags.paddle.NECK_X1, paddle_tags.paddle.RIB_X1)
    assert to_rib - back / 2 >= paddle_tags.BACK_GAP - 1e-9
    assert lay["faces"]["front"]["edge"] >= 90 and back >= 85
