"""Which way does each joint turn? Ask gravity.

    .venv/bin/python -m hw.gravity_sign

A position servo settles a little downhill of its goal. Approaching the same
goal from above and from below and averaging the two settled errors cancels
friction and leaves the sag, whose direction - in ticks - is compared with the
direction the sim says gravity pulls that joint, under both possible signs.
The sign under which the sim's pose agrees is the joint's direction. Needs a
pose where gravity actually loads the joint, so it is run at several.
"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from . import feetech as ft
from .so101 import IDS, JOINTS, SO101

APPROACH = 40  # ticks either side of the goal


def settle(arm, sid, goal, frm):
    arm.bus.write(sid, ft.GOAL_POSITION, int(frm))
    time.sleep(0.6)
    arm.bus.write(sid, ft.GOAL_POSITION, int(goal))
    time.sleep(0.8)
    return np.mean([arm.bus.read_retry(sid, ft.POSITION) for _ in range(5)]) - goal


def gravity_dir(arm, q):
    """Direction (+1/-1 in q) gravity pulls each joint at q, from the sim."""
    w = arm.check.world
    m, d = w.model, w.data
    d.qpos[:] = 0
    d.qpos[w.qpos_adr] = q
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    hold = d.qfrc_bias[w.qvel_adr].copy()  # torque needed to hold; gravity pulls the other way
    return -np.sign(hold), np.abs(hold)


def main() -> int:
    arm = SO101()
    arm.configure(acceleration=40, torque_limit=600)
    arm.torque(True)
    poses = [arm.fold_path[1].copy(), arm.fold_path[2].copy()]
    try:
        arm.unfold()
        for q in poses:
            arm.move_to(q, speed=0.25, check=False)
            t = arm.ticks()
            sag = np.zeros(5)
            for j in (1, 2, 3):
                up = settle(arm, IDS[j], t[j], t[j] + APPROACH)
                dn = settle(arm, IDS[j], t[j], t[j] - APPROACH)
                sag[j] = (up + dn) / 2
            print(f"pose {np.round(arm.q(), 2)}  sag ticks {np.round(sag, 1)}", flush=True)
            for s_name, flip in (("sign +1", 1), ("sign -1", -1)):
                qs = arm.q().copy()
                qs = flip * (arm.ticks() - arm.center) * (2 * np.pi / 4096)
                qs[[0, 4]] = arm.q()[[0, 4]]
                gdir, mag = gravity_dir(arm, qs)
                # sag in ticks predicted: gravity direction in q, times the sign
                pred = gdir * flip
                print(f"  {s_name}: sim gravity N.m {np.round(mag[1:4], 2)}  predicts sag sign "
                      f"{pred[1:4].astype(int)}  measured {np.sign(np.round(sag[1:4])).astype(int)}", flush=True)
    finally:
        arm.hold()
        arm.fold()
        arm.torque(False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
