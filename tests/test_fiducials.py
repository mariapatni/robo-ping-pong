"""The STag generator is a port, so it is tested against the real detector.

`stag-python` detects but does not generate; upstream ships markers as an image
download. sim/fiducials.py is therefore a port of the reference generator's
geometry, and the only test worth having is the round trip: render a marker,
hand it to the actual STag detector, and require the right id back. If the port
drifts, this fails.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.config import MarkerConfig  # noqa: E402
from sim.fiducials import N_BITS, code_locations, codewords, marker  # noqa: E402

LIBRARY_SIZES = {19: 38, 21: 12, 23: 6}


@pytest.mark.parametrize("hd,size", LIBRARY_SIZES.items())
def test_library_sizes_match_upstream(hd, size):
    words = codewords(hd)
    assert len(words) == size
    assert all(len(w) == N_BITS for w in words)


def test_bit_locations_are_inside_the_inner_circle():
    locs = code_locations()
    assert locs.shape == (N_BITS, 2)
    # Centred on (0.5, 0.5) in inner-circle units, none outside the disc.
    assert np.all(np.linalg.norm(locs - 0.5, axis=1) <= 0.5)
    # Four-fold symmetry: the set is 12 positions replicated at 90 degrees.
    assert len({tuple(np.round(p, 6)) for p in locs}) == N_BITS


def test_markers_differ_from_each_other():
    a, b = marker(0, hd=21, size=128), marker(1, hd=21, size=128)
    assert np.abs(a.astype(int) - b.astype(int)).mean() > 5


@pytest.mark.parametrize("marker_id", range(12))
def test_generated_markers_decode_to_their_own_id(marker_id):
    """The port is correct iff the real detector agrees. HD21 is what the
    corners use, so every one of its 12 markers is checked."""
    cv2 = pytest.importorskip("cv2")
    stag = pytest.importorskip("stag")

    img = marker(marker_id, hd=21, size=600)
    page = np.full((760, 760), 255, np.uint8)
    page[80:680, 80:680] = img

    _, ids, _ = stag.detectMarkers(cv2.cvtColor(page, cv2.COLOR_GRAY2BGR), 21)
    assert ids is not None and len(ids), f"marker {marker_id} was not detected at all"
    assert int(np.ravel(ids)[0]) == marker_id


def test_the_corner_ids_are_four_distinct_markers_in_the_library():
    cfg = MarkerConfig()
    assert len(set(cfg.ids)) == 4
    assert max(cfg.ids) < len(codewords(cfg.hd))
