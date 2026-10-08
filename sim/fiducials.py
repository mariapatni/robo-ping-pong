"""STag fiducial markers, generated rather than downloaded.

STag (Benligiray, Topal, Akinlar 2019) uses a square outer border for
detection and an inner *circular* border to refine the homography, which is
what makes its pose estimates steadier than a plain square marker like ArUco.
That stability is the reason to use it here: the markers define the table
corners, and every millimetre of error in the table frame propagates straight
into the ball position the policy sees.

The upstream project ships markers as image downloads, not as a library call,
so this is a port of the reference generator's geometry
(`ref/marker_generator/Form1.cs` in ManfredStoiber/stag). Every constant below
is from that file. The port is validated the only way that means anything:
`tests/test_fiducials.py` renders each marker and feeds it to the real STag
detector, which must recover the right id.

The morphological dilate/erode/smooth passes of the original are skipped; they
are edge cosmetics, and supersampling gives cleaner antialiasing. The detector
round-trip is what proves that is safe.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

# -- geometry, as ratios of the marker edge (Form1.cs lines 20-40) ------------
N_BITS = 48
BORDER = 0.125
OUTER_CIRCLE_R = 0.40
INNER_CIRCLE_R = 0.35
CODE_R = 0.062482177287080  # relative to the inner circle diameter
FILLER_R = 0.7  # relative to CODE_R

LIBRARY_DIR = Path(__file__).resolve().parent / "models" / "stag"

# The 12 bit positions of one quadrant, as (radius, angle) in the inner circle.
_QUADRANT = (
    (0.088363142525988, 0.785398163397448),
    (0.206935928182607, 0.459275804122858),
    (0.206935928182607, np.pi / 2 - 0.459275804122858),
    (0.313672146827381, 0.200579720495241),
    (0.327493143484516, 0.591687617505840),
    (0.327493143484516, np.pi / 2 - 0.591687617505840),
    (0.313672146827381, np.pi / 2 - 0.200579720495241),
    (0.437421957035861, 0.145724938287167),
    (0.437226762361658, 0.433363129825345),
    (0.430628029742607, 0.785398163397448),
    (0.437226762361658, np.pi / 2 - 0.433363129825345),
    (0.437421957035861, np.pi / 2 - 0.145724938287167),
)


@lru_cache(maxsize=1)
def code_locations() -> np.ndarray:
    """The 48 bit centres, in [0,1] coordinates of the inner circle's box."""
    locs = []
    for quadrant in range(4):
        for radius, angle in _QUADRANT:
            a = angle + quadrant * (np.pi / 2)
            locs.append((0.5 + np.cos(a) * radius, 0.5 - np.sin(a) * radius))
    return np.array(locs)


@lru_cache(maxsize=8)
def codewords(hd: int) -> tuple[str, ...]:
    """The codeword library for a given Hamming distance."""
    path = LIBRARY_DIR / f"HD{hd}.txt"
    if not path.exists():
        raise FileNotFoundError(f"no codeword library at {path}")
    words = tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
    bad = [w for w in words if len(w) != N_BITS or set(w) - {"0", "1"}]
    if bad:
        raise ValueError(f"{path} holds malformed codewords")
    return words


def marker(marker_id: int, hd: int = 21, size: int = 512, supersample: int = 4) -> np.ndarray:
    """Render one STag marker as a uint8 greyscale image, white background.

    Drawing order follows the reference generator exactly: black square, white
    outer disc, black circles for set bits, black filler circles bridging
    nearby set bits, white circles for clear bits, then the ring between the
    inner and outer circle forced back to white.
    """
    import cv2

    words = codewords(hd)
    if not 0 <= marker_id < len(words):
        raise IndexError(f"HD{hd} has {len(words)} markers; asked for id {marker_id}")
    bits = np.array([int(c) for c in words[marker_id]])

    s = size * supersample
    img = np.full((s, s), 255, np.uint8)

    marker_px = s / (1 + 2 * BORDER)
    border_px = marker_px * BORDER
    outer_d = marker_px * OUTER_CIRCLE_R * 2
    inner_d = marker_px * INNER_CIRCLE_R * 2
    inner_tl = border_px + marker_px * (0.5 - INNER_CIRCLE_R)
    code_d = inner_d * CODE_R * 2
    filler_d = code_d * FILLER_R
    centre = (int(round(s / 2)), int(round(s / 2)))

    def disc(cx, cy, diameter, colour):
        cv2.circle(img, (int(round(cx)), int(round(cy))), max(1, int(round(diameter / 2))),
                   int(colour), -1, cv2.LINE_AA)

    cv2.rectangle(img, (int(round(border_px)), int(round(border_px))),
                  (int(round(border_px + marker_px)), int(round(border_px + marker_px))), 0, -1)
    disc(*centre, outer_d, 255)

    locs = code_locations()
    px = inner_tl + inner_d * locs  # bit centres in pixels

    for j in np.flatnonzero(bits == 1):
        disc(px[j, 0], px[j, 1], code_d, 0)

    # Filler circles bridge pairs of set bits that sit close together, so the
    # pattern reads as connected blobs rather than a dotted grid.
    near = np.linalg.norm(locs[:, None, :] - locs[None, :, :], axis=-1) < CODE_R * 4
    for j, k in zip(*np.triu_indices(N_BITS, k=1)):
        if bits[j] and bits[k] and near[j, k]:
            mid = (px[j] + px[k]) / 2
            disc(mid[0], mid[1], filler_d, 0)

    for j in np.flatnonzero(bits == 0):
        disc(px[j, 0], px[j, 1], code_d, 255)

    # Force the annulus between the inner and outer circle white: it is the
    # circular border the detector refines its homography against.
    yy, xx = np.mgrid[0:s, 0:s]
    r = np.hypot(xx - centre[0], yy - centre[1])
    img[(r <= outer_d / 2) & (r >= inner_d / 2)] = 255

    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
