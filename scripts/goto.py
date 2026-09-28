"""Command the arm to a joint-space target.

  Headless (fast, prints the result):
      .venv/bin/python scripts/goto.py --deg 30 -40 50 20 0

  With the viewer (macOS needs mjpython, not python):
      .venv/bin/mjpython scripts/goto.py --deg 30 -40 50 20 0 --view
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.arm import ARM_JOINTS, HOME, Arm  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--deg", nargs=5, type=float, metavar=("PAN", "LIFT", "ELBOW", "WRIST", "ROLL"),
                   help="target joint angles in degrees")
    g.add_argument("--rad", nargs=5, type=float, help="target joint angles in radians")
    p.add_argument("--view", action="store_true", help="open the interactive viewer (needs mjpython on macOS)")
    p.add_argument("--duration", type=float, default=1.5, help="seconds to ramp the target over")
    return p.parse_args(argv)


def target_from(args) -> np.ndarray:
    if args.deg is not None:
        return np.radians(args.deg)
    if args.rad is not None:
        return np.array(args.rad, dtype=float)
    return HOME.copy()


def report(arm: Arm, target: np.ndarray) -> None:
    err = np.degrees(np.abs(arm.qpos - target))
    width = max(len(n) for n in ARM_JOINTS)
    print(f"\n{'joint':<{width}}  {'target':>8}  {'reached':>8}  {'error':>7}")
    for i, name in enumerate(ARM_JOINTS):
        print(f"{name:<{width}}  {np.degrees(target[i]):>7.2f}d  {np.degrees(arm.qpos[i]):>7.2f}d  {err[i]:>6.3f}d")
    print(f"\nmax joint error : {err.max():.3f} deg")
    print(f"tip position    : {np.round(arm.tip_pos(), 4)} m")


def ramp(arm: Arm, start: np.ndarray, target: np.ndarray, duration: float, on_step=None) -> None:
    """Walk the position target from `start` to `target` over `duration`.

    This is a placeholder for the jerk-limited blend described in the plan; a
    linear ramp is enough to keep the servos from being asked for a step change.
    """
    n = max(1, int(duration * 50))
    for k in range(1, n + 1):
        arm.command(start + (target - start) * (k / n))
        arm.step(arm.steps_per_command)
        if on_step is not None:
            on_step()


def main(argv=None) -> int:
    args = parse_args(argv)
    target = target_from(args)

    arm = Arm()
    clipped = arm.clip(target)
    if not np.allclose(clipped, target):
        print("note: target clipped to joint limits", file=sys.stderr)
        target = clipped

    if not args.view:
        ramp(arm, arm.qpos, target, args.duration)
        info = arm.goto(target)
        report(arm, target)
        print(f"settled         : {info['settled']}")
        return 0

    import mujoco.viewer

    with mujoco.viewer.launch_passive(arm.model, arm.data) as viewer:
        wall = time.time()

        def sync():
            nonlocal wall
            viewer.sync()
            wall += arm.model.opt.timestep * arm.steps_per_command
            delay = wall - time.time()
            if delay > 0:
                time.sleep(delay)

        ramp(arm, arm.qpos, target, args.duration, on_step=sync)
        report(arm, target)
        print("\nviewer open - close the window to exit")
        while viewer.is_running():
            arm.command(target)
            arm.step(arm.steps_per_command)
            sync()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
