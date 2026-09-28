"""Cycle the arm through a few joint-space waypoints, in the viewer.

    .venv/bin/mjpython scripts/demo.py

macOS requires mjpython (not python) for any MuJoCo viewer window.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.arm import HOME, Arm  # noqa: E402

WAYPOINTS_DEG = [
    (0, -35, 50, 30, 0),
    (45, -55, 70, 25, 40),
    (-45, -55, 70, 25, -40),
    (0, -80, 90, 40, 0),
    (0, 20, -20, -40, 0),
]


def main() -> int:
    import mujoco.viewer

    arm = Arm()
    waypoints = [arm.clip(np.radians(w)) for w in WAYPOINTS_DEG] + [HOME]

    with mujoco.viewer.launch_passive(arm.model, arm.data) as viewer:
        wall = time.time()
        start = arm.qpos
        idx = 0
        print("cycling waypoints - close the window to stop")

        while viewer.is_running():
            target = waypoints[idx % len(waypoints)]
            steps = int(1.5 * 50)
            for k in range(1, steps + 1):
                if not viewer.is_running():
                    break
                arm.command(start + (target - start) * (k / steps))
                arm.step(arm.steps_per_command)
                viewer.sync()
                wall += arm.model.opt.timestep * arm.steps_per_command
                delay = wall - time.time()
                if delay > 0:
                    time.sleep(delay)
            err = np.degrees(np.abs(arm.qpos - target)).max()
            print(f"waypoint {idx % len(waypoints)}  max error {err:5.2f} deg  tip {np.round(arm.tip_pos(), 3)}")
            start = arm.qpos
            idx += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
