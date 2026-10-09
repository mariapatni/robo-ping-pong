"""A plan view of where each arm can legally hit, and where it must aim.

    .venv/bin/python scripts/zone_map.py      # -> out/zone_map.png

Not a render of the simulator: a to-scale top-down drawing, because the
question "can the arm cover its half" is a 2D question and a 3D view answers it
badly. Two panels compare the arm bolted at the back edge against the arm moved
forward onto the table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import replace  # noqa: E402

from sim.config import EnvConfig, MarkerConfig, TableConfig  # noqa: E402
from sim.env import PingPongEnv  # noqa: E402
from sim.paddle import PaddleConfig  # noqa: E402
from sim.zones import grid, sweep_face, zones  # noqa: E402

PX_PER_M = 520
CELL = 0.02
MARGIN = 54
PANEL_GAP = 56

WOOD = (168, 196, 214)
EDGE = (70, 70, 70)
A_ZONE = (90, 180, 90)
B_ZONE = (200, 140, 70)
TARGET = (60, 110, 235)
INK = (28, 28, 28)
GREY = (120, 120, 120)


def blend(img, mask, colour, xe, ye, alpha, origin):
    ox, oy = origin
    overlay = img.copy()
    for ix, iy in zip(*np.nonzero(mask)):
        x0 = int(ox + xe[ix] * PX_PER_M)
        x1 = int(ox + xe[ix + 1] * PX_PER_M)
        y0 = int(oy + (ye[iy] + abs(ye[0])) * PX_PER_M)
        y1 = int(oy + (ye[iy + 1] + abs(ye[0])) * PX_PER_M)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), colour, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def panel(img, origin, table, z, base_x, title, subtitle):
    ox, oy = origin
    w = int(table.length * PX_PER_M)
    h = int(table.width * PX_PER_M)
    xe, ye = grid(table, CELL)

    cv2.rectangle(img, (ox, oy), (ox + w, oy + h), WOOD, -1)
    # A's own strike zone, and the zone it must land in - which is simply B's
    # strike zone on the far side of the net. B's full zone is not drawn: past
    # the net the two are the same thing, and overlaying both made mud.
    blend(img, z["underhand"], A_ZONE, xe, ye, 0.60, origin)
    blend(img, z["target"], TARGET, xe, ye, 0.42, origin)
    cv2.rectangle(img, (ox, oy), (ox + w, oy + h), EDGE, 2)

    net_x = int(ox + table.half_length * PX_PER_M)
    cv2.line(img, (net_x, oy), (net_x, oy + h), INK, 2)

    m = MarkerConfig()
    for sx in (m.inset, table.length - m.inset):
        for sy in (-table.half_width + m.inset, table.half_width - m.inset):
            c = (int(ox + sx * PX_PER_M), int(oy + (sy + table.half_width) * PX_PER_M))
            cv2.rectangle(img, (c[0] - 7, c[1] - 7), (c[0] + 7, c[1] + 7), INK, -1)
            cv2.rectangle(img, (c[0] - 3, c[1] - 3), (c[0] + 3, c[1] + 3), (245, 245, 245), -1)

    for bx, lbl in ((base_x, "A"), (table.length - base_x, "B")):
        c = (int(ox + bx * PX_PER_M), int(oy + table.half_width * PX_PER_M))
        cv2.circle(img, c, 11, INK, -1)
        cv2.putText(img, lbl, (c[0] - 6, c[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(img, title, (ox, oy - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.78, INK, 2, cv2.LINE_AA)
    cv2.putText(img, subtitle, (ox, oy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.52, GREY, 1, cv2.LINE_AA)
    return w, h


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("out/zone_map.png"))
    p.add_argument("--forward", type=float, default=0.15)
    args = p.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    table = TableConfig()
    world = PingPongEnv(cfg=EnvConfig(paddle=PaddleConfig()))
    world.tip_site = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_SITE, "paddle_face")
    swept = sweep_face(world)

    w = int(table.length * PX_PER_M)
    h = int(table.width * PX_PER_M)
    img = np.full((2 * h + 3 * MARGIN + PANEL_GAP + 128, w + 2 * MARGIN, 3), 248, np.uint8)

    half_cells = int(round(table.half_length / CELL)) * int(round(table.width / CELL))
    rows = []
    for i, bx in enumerate((0.0, args.forward)):
        tbl = replace(table, arm_inset=bx)
        z = zones(world, tbl, cell=CELL, swept=swept)
        own = z["underhand"][: int(round(tbl.half_length / CELL))]
        pct = own.sum() / half_cells * 100
        rows.append((bx, pct, z["target"].sum()))
        oy = MARGIN + 46 + i * (h + PANEL_GAP)
        panel(img, (MARGIN, oy), tbl, z, bx,
              f"Arm base at x = {bx:.2f} m" + ("  (as built)" if bx == 0 else "  (moved forward)"),
              f"underhand coverage of its own half: {pct:.1f}%")

    y = img.shape[0] - 60
    for i, (text, colour) in enumerate((
        ("A can strike here, underhand - feeds are aimed across all of it", A_ZONE),
        ("where A must land the return: past the net AND inside B's strike zone", TARGET),
    )):
        yy = y + i * 26
        cv2.rectangle(img, (MARGIN, yy - 12), (MARGIN + 22, yy + 4), colour, -1)
        cv2.putText(img, text, (MARGIN + 32, yy + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.54,
                    INK, 1, cv2.LINE_AA)

    cv2.imwrite(str(args.out), img)
    print(f"{'base x':>7} {'own-half underhand':>20} {'legal target cells':>20}")
    for bx, pct, tgt in rows:
        print(f"{bx:7.2f} {pct:19.1f}% {tgt:20,}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
