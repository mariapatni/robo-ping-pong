"""Animate randomised feeds filling the arm's strike zone.

    .venv/bin/python scripts/feed_spray.py       # -> out/feed_spray.mp4

A top-down plan view: each feed launches from a randomised release point and is
solved to arrive at a strike point sampled uniformly over the underhand zone.
The trajectories are drawn as they fly and the strike points accumulate, so the
coverage of the training distribution is visible rather than asserted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.config import EnvConfig, TableConfig  # noqa: E402
from sim.env import PingPongEnv  # noqa: E402
from sim.feed import _fly, aim_feed, sample_targets  # noqa: E402
from sim.paddle import PaddleConfig  # noqa: E402
from sim.zones import grid, sweep_face, zones  # noqa: E402

PX = 560
MARGIN = 60
TOP = 92
WOOD = (168, 196, 214)
A_ZONE = (90, 180, 90)
TARGET = (60, 110, 235)
INK = (28, 28, 28)
TRAIL = (40, 40, 200)
BALL = (150, 40, 20)  # BGR; renders dark blue once converted for the writer
GREY = (120, 120, 120)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("out/feed_spray.mp4"))
    p.add_argument("--feeds", type=int, default=34)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    table = TableConfig()
    cfg = EnvConfig(paddle=PaddleConfig())
    world = PingPongEnv(cfg=cfg)
    world.tip_site = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_SITE, "paddle_face")
    z = zones(world, table, swept=sweep_face(world))
    xe, ye = grid(table)

    targets = sample_targets(z["underhand"], xe, ye, args.feeds, seed=11)
    feeds = [
        aim_feed(t, cfg.ball, spawn_x=(table.x_far - 0.25, table.x_far + 0.05),
                 table_e=table.restitution, seed=i)
        for i, t in enumerate(targets)
    ]

    w = int(table.length * PX)
    h = int(table.width * PX)

    def to_px(x, y):
        return (int(MARGIN + (x - table.x_near) * PX), int(TOP + (y + table.half_width) * PX))

    # h264 needs even dimensions, and the table's aspect ratio does not care
    # what is convenient, so round the canvas up.
    cw = (w + 2 * MARGIN + 1) // 2 * 2
    ch = (h + TOP + 86 + 1) // 2 * 2
    base = np.full((ch, cw, 3), 248, np.uint8)
    cv2.rectangle(base, to_px(table.x_near, -table.half_width),
                  to_px(table.x_far, table.half_width), WOOD, -1)
    overlay = base.copy()
    for occ, colour in ((z["underhand"], A_ZONE), (z["target"], TARGET)):
        for ix, iy in zip(*np.nonzero(occ)):
            cv2.rectangle(overlay, to_px(xe[ix], ye[iy]), to_px(xe[ix + 1], ye[iy + 1]), colour, -1)
    cv2.addWeighted(overlay, 0.5, base, 0.5, 0, base)
    cv2.rectangle(base, to_px(table.x_near, -table.half_width),
                  to_px(table.x_far, table.half_width), INK, 2)
    cv2.line(base, to_px(table.x_net, -table.half_width), to_px(table.x_net, table.half_width), INK, 2)
    for bx, lbl in ((0.0, "A"), (table.x_far - table.arm_inset, "B")):
        c = to_px(bx, 0.0)
        cv2.circle(base, c, 12, INK, -1)
        cv2.putText(base, lbl, (c[0] - 6, c[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255), 1, cv2.LINE_AA)

    frames = []
    strikes: list[tuple[int, int]] = []
    spawns: list[tuple[int, int]] = []

    for n, f in enumerate(feeds, 1):
        path, dt, bounce = _fly(f.pos, f.vel, cfg.ball.drag_k, cfg.ball.mass,
                                cfg.ball.radius, table.restitution)
        stop = int(f.flight / dt)
        pts = path[: max(2, stop)][::6]
        spawns.append(to_px(f.pos[0], f.pos[1]))
        for k in range(len(pts)):
            img = base.copy()
            for s in spawns:
                cv2.circle(img, s, 4, GREY, -1)
            for s in strikes:
                cv2.circle(img, s, 7, BALL, -1)
                cv2.circle(img, s, 7, (255, 255, 255), 1)
            trail = [to_px(q[0], q[1]) for q in pts[: k + 1]]
            for a, b in zip(trail, trail[1:]):
                cv2.line(img, a, b, TRAIL, 2, cv2.LINE_AA)
            cv2.circle(img, trail[-1], 8, BALL, -1)
            cv2.putText(img, "Randomised feeds, each solved to land in the strike zone",
                        (MARGIN, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.74, INK, 2, cv2.LINE_AA)
            cv2.putText(img, f"feed {n} of {len(feeds)}   release points grey, strike points blue",
                        (MARGIN, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.52, GREY, 1, cv2.LINE_AA)
            y = img.shape[0] - 28
            cv2.rectangle(img, (MARGIN, y - 12), (MARGIN + 22, y + 4), A_ZONE, -1)
            cv2.putText(img, "where A may strike, underhand", (MARGIN + 30, y + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1, cv2.LINE_AA)
            cv2.rectangle(img, (MARGIN + 390, y - 12), (MARGIN + 412, y + 4), TARGET, -1)
            cv2.putText(img, "where A must land its return", (MARGIN + 420, y + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1, cv2.LINE_AA)
            frames.append(img)
        strikes.append(to_px(f.target[0], f.target[1]))

    for _ in range(args.fps):
        frames.append(frames[-1])
    # Drawn with OpenCV, which is BGR; imageio writes RGB.
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    imageio.mimsave(args.out, rgb, fps=args.fps, macro_block_size=1)
    print(f"{len(feeds)} feeds, {len(frames)} frames -> {args.out}")
    tg = np.array([f.target for f in feeds])
    print(f"strike points span x {tg[:,0].min():+.2f}..{tg[:,0].max():+.2f}, y {tg[:,1].min():+.2f}..{tg[:,1].max():+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
