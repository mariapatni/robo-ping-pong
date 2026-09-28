"""Render the arm moving through waypoints to a video file.

No window, no mjpython, no Xcode license needed - this uses MuJoCo's offscreen
renderer. It is also how we will review training rollouts later, which is the
single best defence against a policy that is quietly gaming the reward.

    .venv/bin/python scripts/render.py
    .venv/bin/python scripts/render.py --out out/demo.mp4 --fps 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim.arm import HOME, Arm  # noqa: E402

from demo import WAYPOINTS_DEG  # noqa: E402


def make_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.10, 0.0, 0.14)
    cam.distance = 0.95
    cam.azimuth = 140.0
    cam.elevation = -18.0
    return cam


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("out/demo.mp4"))
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--seconds-per-move", type=float, default=1.2)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    arm = Arm(offscreen=(args.width, args.height))
    cam = make_camera()
    waypoints = [arm.clip(np.radians(w)) for w in WAYPOINTS_DEG] + [arm.clip(HOME)]

    # Render every Nth control step so the video plays at wall-clock speed.
    control_dt = arm.model.opt.timestep * arm.steps_per_command
    steps_per_frame = max(1, round(1.0 / (args.fps * control_dt)))

    frames = []
    with mujoco.Renderer(arm.model, height=args.height, width=args.width) as renderer:
        start = arm.qpos
        for i, target in enumerate(waypoints):
            n = max(1, int(args.seconds_per_move / control_dt))
            for k in range(1, n + 1):
                arm.command(start + (target - start) * (k / n))
                arm.step(arm.steps_per_command)
                if k % steps_per_frame == 0:
                    renderer.update_scene(arm.data, camera=cam)
                    frames.append(renderer.render())
            err = np.degrees(np.abs(arm.qpos - target)).max()
            print(f"waypoint {i}  max error {err:5.2f} deg  tip {np.round(arm.tip_pos(), 3)}")
            start = arm.qpos

    imageio.mimsave(args.out, frames, fps=args.fps, macro_block_size=1)
    print(f"\nwrote {args.out}  ({len(frames)} frames, {len(frames)/args.fps:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
