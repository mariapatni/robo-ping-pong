"""Render the arm with the paddle fitted, offscreen.

Stills by default, because a still is what a design review can actually look at,
and a short sweep of the wrist if you ask for it. No window, no mjpython.

    .venv/bin/python scripts/render_paddle.py
    .venv/bin/python scripts/render_paddle.py --video out/paddle_sweep.mp4

Writes cad/out/paddle_fitted.png - next to the CAD, not in out/, because out/ is
gitignored and this is the design-review image. A 2x2 contact sheet: the whole
arm, the mount from two sides, and the hitting face square on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.arm import Arm  # noqa: E402
from sim.config import TableConfig  # noqa: E402
from sim.paddle import PADDLE_HOME, PaddleConfig, add_paddle, face_pose  # noqa: E402
from sim.table import add_table  # noqa: E402

# (label, lookat, distance, azimuth, elevation) - the four views of the sheet.
VIEWS = (
    ("arm and paddle", (0.16, -0.02, 0.12), 0.80, 128.0, -14.0),
    ("hub on the horn, outboard", (0.18, -0.005, 0.235), 0.13, 75.0, -10.0),
    ("the four screw heads, up the roll axis", (0.172, 0.0, 0.244), 0.09, 205.0, 45.0),
    ("hitting face", (0.25, -0.02, 0.12), 0.28, 195.0, -4.0),
)


def build_arm(width: int, height: int, paddle: PaddleConfig, with_table: bool) -> Arm:
    table = TableConfig()

    def modify(spec: mujoco.MjSpec) -> None:
        if with_table:
            add_table(spec, table)
        add_paddle(spec, paddle)

    return Arm(offscreen=(width, height), modify=modify)


def camera(lookat, distance, azimuth, elevation) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def framed(img: np.ndarray) -> np.ndarray:
    """A dark border, so the four tiles read as four pictures and not one.

    No captions: that would mean a font dependency for the sake of four labels,
    and VIEWS above is the legend in a fixed order.
    """
    out = img.copy()
    out[:3, :, :] = out[-3:, :, :] = out[:, :3, :] = out[:, -3:, :] = (38, 38, 46)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("cad/out/paddle_fitted.png"))
    p.add_argument("--video", type=Path, default=None, help="also sweep wrist_roll to this mp4")
    p.add_argument("--width", type=int, default=560)
    p.add_argument("--height", type=int, default=420)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-table", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    cfg = PaddleConfig()
    arm = build_arm(args.width, args.height, cfg, not args.no_table)
    arm.reset(PADDLE_HOME)

    pos, normal = face_pose(arm.model, arm.data)
    print(f"paddle mass      {cfg.mass*1e3:.1f} g  ({cfg.plastic_mass*1e3:.1f} g printed)")
    print(f"gripper frame    {np.round(arm.tip_pos(), 4)}")
    print(f"paddle face      {np.round(pos, 4)}  normal {np.round(normal, 3)}")
    print(f"face beyond tip  {np.linalg.norm(pos - arm.tip_pos())*1e3:.0f} mm")
    print(f"contacts at PADDLE_HOME {arm.data.ncon}")

    tiles = []
    with mujoco.Renderer(arm.model, height=args.height, width=args.width) as r:
        for _name, lookat, dist, az, el in VIEWS:
            r.update_scene(arm.data, camera=camera(lookat, dist, az, el))
            tiles.append(framed(r.render()))
    sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    imageio.imwrite(args.out, sheet)
    print(f"wrote {args.out}  ({sheet.shape[1]}x{sheet.shape[0]})")

    if args.video is not None:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        frames = []
        cam = camera(*VIEWS[0][1:])
        with mujoco.Renderer(arm.model, height=args.height, width=args.width) as r:
            for roll in np.linspace(-1.4, 1.4, 90):
                q = PADDLE_HOME.copy()
                q[4] = roll
                arm.command(q)
                arm.step(arm.steps_per_command)
                r.update_scene(arm.data, camera=cam)
                frames.append(r.render())
        imageio.mimsave(args.video, frames, fps=args.fps, macro_block_size=1)
        print(f"wrote {args.video}  ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
