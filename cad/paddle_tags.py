"""The 150 mm paddle with a STag marker inlaid in each face, for a two-colour print.

    .venv/bin/python cad/paddle_tags.py

    cad/out/paddle_tags.3mf          black body + white inlays: open in Bambu Studio
    cad/out/paddle_tags_plain.3mf    the same two parts, the plainest 3MF (if the one above will not open)
    cad/out/paddle_tags_black.stl    the same parts as separate meshes
    cad/out/paddle_tags_white.stl
    cad/out/paddle_tags.json         where each marker is, for the hand-eye calibration
    cad/out/paddle_tags_faces.png    both faces as the camera sees them
    sim/models/paddle_tags/          the two meshes again, for the sim (sim.paddle.tagged_paddle)

The camera calibration needs something on the arm whose 3D corners it can
find to a fraction of a pixel. These are STag HD21 markers (the family the
repo already generates and validates, sim/fiducials.py): id 0 on the hitting
face, id 1 on the back. The paddle is the 150 mm blade (cad/paddle.py --big),
3.0 mm thick so each face can carry an inlay over a black core, with no rubber
- it would cover the marker.

SIZES (all in mm; the blade is 150 across)
  white disc        146 across: a 2 mm black rim, clear of the back chamfer
  hitting face      94 mm black square - the most a 146 mm disc holds with
                    TAG_CORNER_CLEAR (6 mm) of white beyond each corner
  back face         88 mm. The neck and rib that join the hub to the blade
                    run 25 mm in from the rim on this side, ending 50 mm from
                    the blade centre; a 94 mm square would come within 3 mm of
                    them - about 3 px at a metre, close enough for the black
                    rib to merge into the marker's edge. 88 mm leaves 6.
  relief            the white stands 0.1 mm proud on the back, 0.08 mm on the
                    hitting face (flattened into the first layer on the bed)
  inlays            white 0.8 mm deep on the hitting face (it prints first,
                    face down: four layers to hide the black over it), 0.6 mm
                    on the back (the top three layers); 1.6 mm of black core
                    between them, so neither face shows through the other

ORIENTATION
  Both markers are drawn with their "up" towards the blade tip (+x). The
  hitting face prints against the bed and is seen from outside, from -z, so it
  is laid out mirrored relative to the back; `tests/test_paddle_tags.py`
  renders each face as the camera will see it and requires the STag detector
  to read the right id.

Print: hitting face down on textured PEI (matte, so no glare), black body,
white inlays, no supports - as cad/paddle.py's notes. Measure each printed
marker's edge with calipers; the calibration takes it as an input.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import paddle  # noqa: E402

BLADE_DIA = 150.0
BLADE_THICK = 3.0
TAG_HD = 21
TAG_IDS = {"front": 0, "back": 1}
WHITE_RIM = 2.0  # black ring left round the white disc
TAG_CORNER_CLEAR = 6.0  # white beyond each corner of the black square
BACK_GAP = 6.0  # white between the end of the neck/rib and the back square
INLAY = {"front": 0.8, "back": 0.6}
# The white stands proud of the black by this much, so each colour is its own
# surface in the slicer. On the back (facing up) 0.1 mm. The hitting face
# prints on the bed, where any step lies inside the first layer (0.2 mm,
# sliced at its middle): 0.08 mm stays below that middle, so both colours
# still print in layer one and the face comes out flat.
RELIEF = {"front": 0.08, "back": 0.1}
PX_PER_MM = 24.0  # how finely the marker image is traced into outlines

OUT = HERE / "out"
SIM_DIR = HERE.parent / "sim" / "models" / "paddle_tags"


def configure() -> None:
    """The 150 mm, 3 mm, rubber-free paddle in cad/paddle.py."""
    paddle.BLADE_THICK = BLADE_THICK
    paddle.GROOVE_DEPTH = 0.0
    paddle.RUBBER_THICK = 0.0
    paddle.set_blade(BLADE_DIA)


def layout() -> dict:
    """Marker sizes and placement, in the paddle frame (x down the roll axis,
    y across the blade, z = 0 the hitting face)."""
    configure()
    r_white = paddle.BLADE_R - WHITE_RIM
    front = np.floor(np.sqrt(2.0) * (r_white - TAG_CORNER_CLEAR))
    # On the back, the neck and rib end at RIB_X1 (the further in of the two).
    to_rib = paddle.BLADE_CX - max(paddle.NECK_X1, paddle.RIB_X1)
    back = min(front, np.floor(2.0 * (to_rib - BACK_GAP)))
    centre = np.array([paddle.BLADE_CX, 0.0])
    return {
        "blade_dia": BLADE_DIA, "blade_thick": BLADE_THICK, "white_r": r_white, "centre_xy": centre.tolist(),
        "faces": {
            # right/up: the marker image's x and -row directions, in the paddle frame,
            # as seen from outside that face.
            # z: the marker's white surface, RELIEF proud of the black.
            "front": {"id": TAG_IDS["front"], "edge": float(front), "z": -RELIEF["front"], "outward": [0, 0, -1],
                      "right": [0, 1, 0], "up": [1, 0, 0], "inlay": INLAY["front"], "relief": RELIEF["front"]},
            "back": {"id": TAG_IDS["back"], "edge": float(back), "z": BLADE_THICK + RELIEF["back"],
                     "outward": [0, 0, 1], "right": [0, -1, 0], "up": [1, 0, 0], "inlay": INLAY["back"],
                     "relief": RELIEF["back"]},
        },
    }


def marker_image(marker_id: int, edge: float) -> tuple[np.ndarray, float]:
    """The marker's black pattern as a bool image (True = black), cropped to
    the black square, and its mm per pixel."""
    from sim.fiducials import BORDER, marker
    n = int(round(edge * PX_PER_MM * (1 + 2 * BORDER)))
    img = marker(marker_id, hd=TAG_HD, size=n, supersample=2)
    b = int(round(n * BORDER / (1 + 2 * BORDER)))
    sq = img[b:n - b, b:n - b] < 128
    return sq, edge / sq.shape[0]


def black_polygons(face: str, lay: dict) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """The marker's black regions on one face, as (outer, holes) polygons in
    the paddle's x, y (mm)."""
    import cv2
    f = lay["faces"][face]
    sq, mm = marker_image(f["id"], f["edge"])
    contours, hier = cv2.findContours(sq.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hier = hier[0]
    centre = np.array(lay["centre_xy"])
    right, up = np.array(f["right"][:2], float), np.array(f["up"][:2], float)
    half = sq.shape[0] / 2.0

    def to_xy(c):
        c = cv2.approxPolyDP(c, 0.35, True).reshape(-1, 2).astype(float)
        # pixel (col, row) -> paddle x, y; pixel edges, so +0.5 is the pixel centre's boundary
        return centre + ((c[:, 0] + 0.5 - half) * mm)[:, None] * right + ((half - c[:, 1] - 0.5) * mm)[:, None] * up

    polys = []
    for i, h in enumerate(hier):
        if h[3] != -1:
            continue  # a hole; collected with its parent
        holes, j = [], h[2]
        while j != -1:
            holes.append(to_xy(contours[j]))
            j = hier[j][0]
        polys.append((to_xy(contours[i]), holes))
    return polys


def face_image(face: str, lay: dict, px_per_mm: float = 6.0) -> np.ndarray:
    """One face as the camera sees it from outside: white disc, black marker,
    black blade around (greyscale). For the orientation test, and the preview."""
    import cv2
    f = lay["faces"][face]
    r = lay["blade_dia"] / 2
    n = int(round(2 * r * px_per_mm)) + 20
    img = np.zeros((n, n), np.uint8)
    c = np.array(lay["centre_xy"])
    # How the face looks from outside follows from which way it faces, not from
    # how the marker was laid out on it: right = up x outward.
    up3 = np.array(f["up"], float)
    right = np.cross(up3, np.array(f["outward"], float))[:2]
    up = up3[:2]

    def to_px(xy):  # paddle x, y -> image col, row as seen from outside
        d = np.asarray(xy) - c
        return np.c_[d @ right, -(d @ up)] * px_per_mm + n / 2

    cv2.circle(img, (n // 2, n // 2), int(round(lay["white_r"] * px_per_mm)), 255, -1, cv2.LINE_AA)
    # Largest first: the code dots sit inside the hole of the black square, and
    # painting that hole white afterwards would erase them.
    polys = sorted(black_polygons(face, lay), key=lambda ph: -cv2.contourArea(ph[0].astype(np.float32)))
    for outer, holes in polys:
        cv2.fillPoly(img, [np.round(to_px(outer)).astype(np.int32)], 0, cv2.LINE_AA)
        for h in holes:
            cv2.fillPoly(img, [np.round(to_px(h)).astype(np.int32)], 255, cv2.LINE_AA)
    return img


def build():
    from build123d import Circle, Plane, Polygon, Pos, Rectangle, extrude

    lay = layout()
    body = paddle.build()
    cx = lay["centre_xy"][0]
    whites = []
    for face, f in lay["faces"].items():
        z0 = -f["relief"] if face == "front" else BLADE_THICK - f["inlay"]
        plane = Plane.XY.offset(z0)
        region = Pos(cx, 0) * Circle(lay["white_r"])
        for outer, holes in black_polygons(face, lay):
            blob = Polygon(*[tuple(p) for p in outer], align=None)
            for h in holes:
                blob = blob - Polygon(*[tuple(p) for p in h], align=None)
            region = region - blob
        if face == "back":
            # Not under the neck and rib, which stand on the back face.
            x1 = max(paddle.NECK_X1, paddle.RIB_X1)
            region = region - Pos(0.5 * (paddle.HUB_THICK + x1), 0) * Rectangle(x1 - paddle.HUB_THICK,
                                                                             paddle.NECK_WIDTH)
        whites.append(extrude(plane * region, amount=f["inlay"] + f["relief"]))
    white = whites[0] + whites[1]
    black = body - white
    return black, white, lay


def main() -> int:
    import cv2
    from build123d import export_stl

    OUT.mkdir(exist_ok=True)
    black, white, lay = build()
    export_stl(black, str(OUT / "paddle_tags_black.stl"), tolerance=paddle.STL_TOL_FINE,
               angular_tolerance=paddle.STL_ANG_FINE)
    export_stl(white, str(OUT / "paddle_tags_white.stl"), tolerance=paddle.STL_TOL_FINE,
               angular_tolerance=paddle.STL_ANG_FINE)
    # From the STLs: build123d's 3MF writer refuses the few stray edges the
    # traced inlays leave (cad/threemf.py says more).
    from threemf import write_3mf, write_plain_3mf
    parts = {n: paddle.read_stl(OUT / f"paddle_tags_{n}.stl") for n in ("black", "white")}
    write_3mf(OUT / "paddle_tags.3mf", parts, name="paddle_tags")
    write_plain_3mf(OUT / "paddle_tags_plain.3mf", parts)  # for a slicer that will not open the one above
    import shutil
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    for n in ("black", "white"):
        shutil.copyfile(OUT / f"paddle_tags_{n}.stl", SIM_DIR / f"{n}.stl")
    (OUT / "paddle_tags.json").write_text(json.dumps(lay, indent=2) + "\n")
    faces = [face_image(f, lay) for f in ("front", "back")]
    cv2.imwrite(str(OUT / "paddle_tags_faces.png"), np.hstack(faces))
    fr, bk = lay["faces"]["front"]["edge"], lay["faces"]["back"]["edge"]
    print(f"paddle {BLADE_DIA:.0f} mm x {BLADE_THICK:.1f} mm: STag HD{TAG_HD} id 0 = {fr:.0f} mm on the hitting "
          f"face, id 1 = {bk:.0f} mm on the back; black {black.volume / 1000:.1f} cm3, white "
          f"{white.volume / 1000:.1f} cm3; wrote {OUT}/paddle_tags.*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
