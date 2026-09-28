"""Render a launched ball crossing the table to a video file.

Same offscreen path as scripts/render.py - no window, no mjpython, no Xcode
licence. This is the only way to actually look at the ball physics on this
machine, so it is worth keeping cheap to run.

    .venv/bin/python scripts/render_rally.py
    .venv/bin/python scripts/render_rally.py --seed 7 --seconds 3 --out out/rally7.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.ball import drag_free_landing  # noqa: E402
from sim.env import PingPongEnv  # noqa: E402


def make_camera(env: PingPongEnv) -> mujoco.MjvCamera:
    """Look down the table from behind and above the arm, so the bounce and the
    depth of the ball are both legible in a flat video."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.45 * env.cfg.table.length, 0.0, 0.05)
    cam.distance = 1.9
    cam.azimuth = 150.0
    cam.elevation = -22.0
    return cam


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("out/rally.mp4"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seconds", type=float, default=2.5)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    env = PingPongEnv(offscreen=(args.width, args.height))
    env.reset()
    feed = env.launch(args.seed)
    print(f"launch  pos {np.round(feed.pos, 3)}  vel {np.round(feed.vel, 3)}  speed {feed.speed:.2f} m/s")
    print(f"drag-free landing estimate: {np.round(drag_free_landing(feed), 3)}")

    dt = env.model.opt.timestep
    steps_per_frame = max(1, round(1.0 / (args.fps * dt)))
    n_frames = int(args.seconds * args.fps)

    frames = np.empty((n_frames, args.height, args.width, 3), dtype=np.uint8)
    with mujoco.Renderer(env.model, height=args.height, width=args.width) as renderer:
        for f in range(n_frames):
            env.step(steps_per_frame)
            renderer.update_scene(env.data, camera=make_camera(env))
            frames[f] = renderer.render()

    print(f"ball ended at {np.round(env.ball_pos(), 3)}  speed {np.linalg.norm(env.ball_vel()):.2f} m/s")
    imageio.mimsave(args.out, list(frames), fps=args.fps, macro_block_size=1)
    print(f"wrote {args.out}  ({n_frames} frames, {args.seconds:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
