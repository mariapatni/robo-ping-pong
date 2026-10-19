"""Calibrate by hand: torque off, a person moves each joint through its range.

    .venv/bin/python -m hw.hand_calibrate            # record, then write the servos
    .venv/bin/python -m hw.hand_calibrate --dry      # record and report only

The automatic rehome (hw/rehome.py) drives joints into their stops, which
only works if the old calibration is good enough to keep the arm off the
table while it does - and on this arm it was not. This is LeRobot's method:
with the torque off, move shoulder pan, shoulder lift, elbow and wrist flex
each slowly from one end of its travel to the other. The extremes are logged
live; a joint is done when its recorded range is at least 85% of the sim's.
It runs until Ctrl-C (or --timeout): finishing on stillness alone would record
a mid-range pause as the rest pose.
Then each joint is centred on its range, exactly as the rehome would.

Fold the arm back to its rest pose before it finishes: that pose is recorded
as the start of the fold path that hw.park and hw.unfold use.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .rehome import log, write_calibration
from .so101 import BACKUP, JOINTS, RAD_PER_TICK, SO101

DONE = 0.85


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry", action="store_true")
    p.add_argument("--timeout", type=float, default=300.0)
    args = p.parse_args(argv)

    arm = SO101(log=log)
    backup = json.loads(BACKUP.read_text())
    fold_ticks = [arm.center + w / RAD_PER_TICK * arm.sign for w in arm.fold_path]
    arm.torque(False)
    need = 2 * arm.check.upper / RAD_PER_TICK * DONE  # ticks of travel per joint
    t = arm.ticks()
    lo, hi = t.copy(), t.copy()
    log("TORQUE OFF. Move shoulder_pan, shoulder_lift, elbow_flex, wrist_flex each through its full range.")
    t0 = time.time()
    last_print = 0.0
    try:
        while time.time() - t0 < args.timeout:
            t = arm.ticks()
            lo, hi = np.minimum(lo, t), np.maximum(hi, t)
            frac = (hi - lo)[:4] / need[:4]
            if time.time() - last_print > 1.0:
                log("  " + ("ALL COVERED - fold it to rest, then stop  " if np.all(frac >= 1) else "") + "  ".join(f"{JOINTS[j][:11]:11s} {lo[j]:4.0f}..{hi[j]:4.0f} {min(frac[j], 9.99):4.0%}"
                                     for j in range(4)))
                last_print = time.time()
            time.sleep(0.02)
    except KeyboardInterrupt:
        log("stopped by hand")

    stops = {j: (float(lo[j]), float(hi[j])) for j in range(4) if (hi[j] - lo[j]) >= need[j]}
    for j in range(4):
        w = np.degrees((hi[j] - lo[j]) * RAD_PER_TICK)
        sim = np.degrees(2 * arm.check.upper[j])
        log(f"{JOINTS[j]}: {lo[j]:.0f}..{hi[j]:.0f} = {w:.0f} deg (sim {sim:.0f}) "
            f"{'ok' if j in stops else 'NOT COVERED - left as it was'}")
    # The pose it was left in becomes the rest of the fold path.
    fold_ticks[0] = arm.ticks()
    (BACKUP.parent / "hand_calibration.json").write_text(json.dumps(
        {JOINTS[j]: {"lo": float(lo[j]), "hi": float(hi[j])} for j in range(5)}, indent=2))
    if args.dry or not stops:
        log("nothing written")
        return 0
    write_calibration(arm, backup, stops, fold_ticks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
