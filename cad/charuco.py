"""The ChArUco board for calibrating the webcam's lens, as a two-colour print.

    .venv/bin/python cad/charuco.py

    cad/out/charuco_board.3mf          two parts, white plate + black pattern: open in Bambu Studio
    cad/out/charuco_board_white.stl    the same two parts as separate meshes
    cad/out/charuco_board_black.stl
    cad/out/charuco_board.json         the board, as hw/calibrate_camera.py needs it
    cad/out/charuco_board.png          300 dpi, for a paper fallback

THE BOARD
---------
4 x 8 squares of 30 mm with 22 mm ArUco markers (DICT_5X5_100, ids 0-15) in
the white squares, in a 4 mm white frame: a 248 x 128 mm plate, half of a
256 mm Bambu bed. Bigger squares are better for a lens calibration - the
corners are found to a fixed fraction of a pixel, so a bigger board at the
same distance pins the lens down more - and 30 mm is the most 8 squares allow
on that bed. Marker cells come out at 22/7 = 3.1 mm, easy for a 0.4 mm nozzle.

PRINTING
--------
Pattern face DOWN on the textured PEI plate: the first layers carry both
colours, and the plate's texture leaves the face matte (a glossy face throws
glare, and glare hides corners). Printed face-down, the face is seen mirrored
from above - so the model is mirrored here, and reads correctly once turned
over. `tests/test_charuco.py` checks that by detecting the board as seen from
the printed face.

The black pattern is BLACK_DEPTH thick, inlaid into a PLATE_THICK white plate.
Chessboard squares touch only at their corners, which as solids is a
non-manifold edge; every black shape is inset by INSET (0.03 mm) so they are
separate islands. The slicer fills a gap that narrow, and the corners the
calibration uses - where square edges cross - do not move.

After printing: glue it to something flat (6 mm glass, a ceramic tile), and
measure 7 squares along each side with calipers. Pass that as the square
size - printers scale, and x and y can differ.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

COLS, ROWS = 8, 4  # squares
SQUARE = 30.0  # mm
MARKER = 22.0  # mm
DICTIONARY = "DICT_5X5_100"
FRAME = 4.0  # white margin around the squares, mm
PLATE_THICK = 3.0
BLACK_DEPTH = 0.6  # three 0.2 mm layers: enough for black to hide the white behind it
INSET = 0.03
# The black stands proud of the white by this much, so each colour is its own
# surface in the slicer. The pattern prints on the bed, where a step lies inside
# the first layer (0.2 mm, sliced at its middle): 0.08 mm stays below that
# middle, so both colours still print in layer one and the face comes out flat.
RELIEF = 0.08

WIDTH = COLS * SQUARE + 2 * FRAME  # 248
HEIGHT = ROWS * SQUARE + 2 * FRAME  # 128

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def board():
    import cv2
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
    return cv2.aruco.CharucoBoard((COLS, ROWS), SQUARE, MARKER, d), d


def black_rects() -> list[tuple[float, float, float, float]]:
    """Every black rectangle (x0, y0, x1, y1) in mm, as the board is SEEN:
    x to the right and y DOWN from the top-left of the plate. Black squares
    where row + column is even (OpenCV's layout); in the white squares the
    markers, ids 0-15 in row order, each a 7 x 7 grid of cells with the
    one-cell black border included."""
    _, d = board()
    rects = []
    cell = MARKER / 7
    pad = (SQUARE - MARKER) / 2
    marker_id = 0
    for r in range(ROWS):
        for c in range(COLS):
            x0, y0 = FRAME + c * SQUARE, FRAME + r * SQUARE
            if (r + c) % 2 == 0:
                rects.append((x0, y0, x0 + SQUARE, y0 + SQUARE))
                continue
            bits = d.generateImageMarker(marker_id, 7) < 128
            marker_id += 1
            for i in range(7):  # merge each row's black cells into runs
                j = 0
                while j < 7:
                    if not bits[i, j]:
                        j += 1
                        continue
                    k = j
                    while k < 7 and bits[i, k]:
                        k += 1
                    rects.append((x0 + pad + j * cell, y0 + pad + i * cell, x0 + pad + k * cell, y0 + pad + (i + 1) * cell))
                    j = k
    return [(a + INSET, b + INSET, c - INSET, e - INSET) for a, b, c, e in rects]


def raster(px_per_mm: float = 8.0, rects=None) -> np.ndarray:
    """The board as seen, as a greyscale image (white plate, black pattern) -
    as it prints: the INSET gaps, far narrower than a nozzle, closed."""
    import cv2
    rects = black_rects() if rects is None else rects
    img = np.full((int(round(HEIGHT * px_per_mm)), int(round(WIDTH * px_per_mm))), 255, np.uint8)
    for x0, y0, x1, y1 in rects:
        x0, y0, x1, y1 = x0 - INSET, y0 - INSET, x1 + INSET, y1 + INSET
        cv2.rectangle(img, (int(round(x0 * px_per_mm)), int(round(y0 * px_per_mm))),
                      (int(round(x1 * px_per_mm)) - 1, int(round(y1 * px_per_mm)) - 1), 0, -1)
    return img


def model_rects() -> list[tuple[float, float, float, float]]:
    """The black rectangles in the CAD frame: x right, y up, z = 0 the printed
    face on the bed. Seen from below - which is how the printed face is seen
    once the board is turned over - x runs the other way, so mirror x here."""
    return [(WIDTH - x1, HEIGHT - y1, WIDTH - x0, HEIGHT - y0) for x0, y0, x1, y1 in black_rects()]


def seen_from_printed_face(px_per_mm: float = 8.0) -> np.ndarray:
    """The board as the camera will see it: the model's z = 0 face, from below.
    Turning the board over about its y axis sends x -> W - x, and image rows
    run down while model y runs up."""
    rects = [(WIDTH - x1, HEIGHT - y1, WIDTH - x0, HEIGHT - y0) for x0, y0, x1, y1 in model_rects()]
    return raster(px_per_mm, rects)


def build():
    from build123d import Box, Compound, Location, Pos

    black = [Pos((x0 + x1) / 2, (y0 + y1) / 2, (BLACK_DEPTH - RELIEF) / 2) * Box(x1 - x0, y1 - y0, BLACK_DEPTH + RELIEF)
             for x0, y0, x1, y1 in model_rects()]
    black = Compound(children=black)
    plate = Location((WIDTH / 2, HEIGHT / 2, PLATE_THICK / 2)) * Box(WIDTH, HEIGHT, PLATE_THICK)
    white = plate - black
    return white, black


def main() -> int:
    import cv2
    from build123d import export_stl

    from threemf import write_3mf
    OUT.mkdir(exist_ok=True)
    white, black = build()
    export_stl(white, str(OUT / "charuco_board_white.stl"))
    export_stl(black, str(OUT / "charuco_board_black.stl"))
    # One object, a part per colour (cad/threemf.py), as the paddle.
    from paddle import read_stl
    parts = {n: read_stl(OUT / f"charuco_board_{n}.stl") for n in ("white", "black")}
    write_3mf(OUT / "charuco_board.3mf", parts, name="charuco_board")
    from threemf import write_plain_3mf
    write_plain_3mf(OUT / "charuco_board_plain.3mf", parts)  # for a slicer that will not open the one above
    spec = {"cols": COLS, "rows": ROWS, "square_mm": SQUARE, "marker_mm": MARKER, "dictionary": DICTIONARY,
            "plate_mm": [WIDTH, HEIGHT, PLATE_THICK], "black_depth_mm": BLACK_DEPTH, "relief_mm": RELIEF,
            "print": "pattern face down on textured PEI; measure 7 squares in x and y after printing"}
    (OUT / "charuco_board.json").write_text(json.dumps(spec, indent=2) + "\n")
    dpi = 300 / 25.4
    cv2.imwrite(str(OUT / "charuco_board.png"), raster(dpi))
    print(f"board {WIDTH:.0f} x {HEIGHT:.0f} x {PLATE_THICK:.0f} mm, {COLS} x {ROWS} squares of {SQUARE:.0f} mm, "
          f"{MARKER:.0f} mm markers ({DICTIONARY}); wrote {OUT}/charuco_board.*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
