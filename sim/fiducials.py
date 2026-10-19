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

The original's morphological passes are ported too, and they are not
cosmetic: the black code circles, the filler circles bridging them and the
white circles for clear bits overlap, and drawn one over another they leave
slivers - thin black crescents inside a white bit, flecks of black between
blobs. The reference then runs five rounds of a majority vote over a disc of
12 px (in its 1000 px image) that turns those into the smooth connected blobs
every published STag has, and redraws the code circles on top so each bit is
an exact circle again. Its final 3x3 edge smoothing is left to the
supersampled render's own antialiasing. The detector round-trip still proves
the result reads.
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

# The reference generator's morphology (Form1.cs drawMarkers): in its 1000 px
# image, five rounds of dilate-then-erode, each a majority vote over a disc of
# radius 12 px.
REF_SIZE = 1000
MORPH_RADIUS = 12
MORPH_ROUNDS = 5

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

    img = _majority_rounds(img, s, outer_d)

    # Redraw the code circles over the cleaned blobs, black then white, so
    # every bit is an exact circle again.
    for j in np.flatnonzero(bits == 1):
        disc(px[j, 0], px[j, 1], code_d, 0)
    for j in np.flatnonzero(bits == 0):
        disc(px[j, 0], px[j, 1], code_d, 255)

    # Force the annulus between the inner and outer circle white: it is the
    # circular border the detector refines its homography against.
    yy, xx = np.mgrid[0:s, 0:s]
    r = np.hypot(xx - centre[0], yy - centre[1])
    img[(r <= outer_d / 2) & (r >= inner_d / 2)] = 255

    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def _majority_rounds(img: np.ndarray, s: int, outer_d: float) -> np.ndarray:
    """Form1.cs dilateBitmap / erodeBitmap, MORPH_ROUNDS times. Dilate: a
    pixel that is not black turns black if more than half its disc is not
    white. Erode: a black pixel turns white if more than half its disc is not
    black. Only inside the outer circle's bounding square, as there. Done on
    the binary image (the reference draws the code circles unantialiased for
    this step), with the disc scaled to this image's size."""
    import cv2

    r = max(1, int(round(MORPH_RADIUS * s / REF_SIZE)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = ((xx * xx + yy * yy) <= r * r).astype(np.float32)
    disc[r, r] = 0.0  # the reference's mask leaves out the centre
    half = disc.sum() / 2.0
    lo = int((s - outer_d) / 2)
    region = np.zeros((s, s), bool)
    region[lo:s - lo, lo:s - lo] = True

    black = img < 128
    for _ in range(MORPH_ROUNDS):
        n_black = cv2.filter2D(black.astype(np.float32), -1, disc, borderType=cv2.BORDER_CONSTANT)
        black = black | (region & (n_black > half))
        n_white = cv2.filter2D((~black).astype(np.float32), -1, disc, borderType=cv2.BORDER_CONSTANT)
        black = black & ~(region & (n_white > half))
    out = img.copy()
    out[black] = 0
    out[~black & region] = 255
    return out
