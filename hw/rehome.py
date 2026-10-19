"""Rehome the SO-101's encoders against its own hard stops.

    .venv/bin/python -m hw.rehome            # find the stops, then write the servos
    .venv/bin/python -m hw.rehome --dry      # find the stops, write nothing

The sim's joint ranges are symmetric about zero, so sim zero is the middle of
each joint's mechanical range. This drives each joint slowly, at reduced
torque, into both of its stops, then sets the homing offset so the mid-point
reads 2048 and the angle limits sit just inside the stops.

Each sweep is made from a staging pose chosen in sim so that nothing but the
joint's own stop can end it (checked again here before every move), and the
measured range is compared against the sim's: a range much shorter than
expected means something other than the stop got in the way, and the run stops
without writing anything.

Wrist roll has no stop - only its cable - so it keeps its current centre.

The old EEPROM values are saved to hw/calibration_backup.json first.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import feetech as ft
from .so101 import CALIBRATION, IDS, JOINTS, RAD_PER_TICK, SO101

STEP = 5  # ticks per 20 ms cycle, ~0.38 rad/s
LAG_STALL = 50  # ticks behind the goal, and ...
STALL_WINDOW = 10  # ... moved less than STALL_MOVE ticks over this many cycles = at the stop.
STALL_MOVE = 4  # Lag alone is not enough: under gravity a joint lags while still moving.
AGREE = 15  # ticks: two stalls this close are the same stop
MARGIN = 25  # ticks (~2 deg) between the angle limit and the stop
TOO_SHORT = 0.85  # measured / sim range below this aborts

# joint index -> staging pose for the - sweep, for the + sweep (rad; None = keep
# the roll the arm started with; the joint's own entry is ignored), torque limit /1000
R = None
SWEEPS = [
    (3, [0, 0, 0, 0, R], [0, 0, 0, 0, R], 300),            # wrist flex: forearm level, paddle in the air
    (2, [0, 0, 0, -1.6, R], [0, 0, 0, -1.6, R], 450),      # elbow: upper arm vertical, paddle tucked
    (0, [0, 0, -1.0, 0, R], [0, 0, -1.0, 0, R], 350),      # pan: forearm angled up, compact
    # lift: elbow bent back; paddle straight going back (table at -2.1 in sim),
    # tucked going forward (table at +1.95). The stops are near +/-1.75.
    (1, [0, 0, -1.6, 0, R], [0, 0, -1.6, -1.6, R], 650),
]



def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def seek(arm: SO101, j: int, direction: int, bound: float, torque_limit: int) -> float:
    """A hard stop is where the joint stalls twice in the same place.

    This arm also stalls at random (see SO101.move_to), so one stall is only a
    candidate: back off and approach again, and accept it when two approaches
    agree to within AGREE ticks. A random stall further along simply becomes
    the start of the next approach."""
    last = None
    for _ in range(6):
        pos = seek_once(arm, j, direction, bound, torque_limit)
        if last is not None and abs(pos - last) <= AGREE:
            return (pos + last) / 2
        last = pos if last is None else (pos if (pos - last) * direction > 0 else last)
        time.sleep(0.3)
    raise RuntimeError(f"{JOINTS[j]}: no repeatable stop in direction {direction:+d}")


def seek_once(arm: SO101, j: int, direction: int, bound: float, torque_limit: int) -> float:
    """Drive joint j toward `direction` until it stalls; return the stall tick."""
    sid = IDS[j]
    bus = arm.bus
    bus.write(sid, ft.TORQUE_LIMIT, torque_limit)
    pos = float(bus.read_retry(sid, ft.POSITION))
    goal, history = pos, []
    try:
        while True:
            goal += direction * STEP
            if (goal - bound) * direction > 0:
                raise RuntimeError(f"{JOINTS[j]}: no stop before {bound:.0f} ticks")
            bus.write(sid, ft.GOAL_POSITION, int(goal))
            time.sleep(0.02)
            pos = float(bus.read_retry(sid, ft.POSITION))
            history.append(pos)
            lag = (goal - pos) * direction
            moved = abs(pos - history[-STALL_WINDOW]) if len(history) >= STALL_WINDOW else np.inf
            if lag > LAG_STALL and moved < STALL_MOVE:
                load = ft.decode_sm(bus.read_retry(sid, ft.LOAD), 10)
                log(f"  {JOINTS[j]} stop {'+' if direction > 0 else '-'} at {pos:.0f} ticks (load {load})")
                return pos
    finally:
        # Back off the stop, then restore full torque.
        bus.write(sid, ft.GOAL_POSITION, int(pos - direction * 80))
        time.sleep(0.3)
        bus.write(sid, ft.TORQUE_LIMIT, int(arm.torque_cap * 1000))


def write_calibration(arm: SO101, backup: dict, stops: dict, fold_ticks) -> None:
    """Centre each joint in `stops` ({index: (lo, hi)} in current ticks) on 2048:
    homing offset so the mid-point reads 2048, angle limits MARGIN inside the
    stops. Joints not in `stops` keep the backup's values. Torque must be off.
    Writes hw/calibration.json, with the fold path re-expressed in the new frame."""
    bus = arm.bus
    cal = {}
    for k, sid in enumerate(IDS):
        name = JOINTS[k]
        old = ft.decode_sm(bus.read_retry(sid, ft.HOMING_OFFSET), 11)
        if k in stops:
            lo, hi = stops[k]
            mid = (lo + hi) / 2
            new = int(round(old + mid - 2048))
            new = (new + 2048) % 4096 - 2048  # fold into the register's range
            half = (hi - lo) / 2 - MARGIN
            mn, mx = int(round(2048 - half)), int(round(2048 + half))
        else:
            new, mn, mx = old, backup[name]["min"], backup[name]["max"]
        before = bus.read_retry(sid, ft.POSITION)
        bus.write(sid, ft.LOCK, 0)
        bus.write(sid, ft.HOMING_OFFSET, ft.encode_sm(new, 11))
        bus.write(sid, ft.MIN_LIMIT, mn)
        bus.write(sid, ft.MAX_LIMIT, mx)
        bus.write(sid, ft.LOCK, 1)
        time.sleep(0.05)
        after = bus.read_retry(sid, ft.POSITION)
        log(f"{name}: offset {old} -> {new}, limits {mn}..{mx}, reading {before} -> {after}")
        if k in stops and abs((before - after) - (mid - 2048)) > 20:
            raise RuntimeError(f"{name}: reading moved {before - after}, expected {mid - 2048:.0f} - "
                               "offset sign is not what this code assumes")
        cal[name] = {"id": sid, "center": 2048, "sign": 1, "min": mn, "max": mx, "offset": new}
        # Later joints' fold ticks are in the old frame of this joint; shift them.
        if k in stops:
            for t in fold_ticks:
                t[k] -= mid - 2048
    arm.center[:] = 2048
    cal["_fold"] = [[round(float(v), 4) for v in arm.to_rad(t)] for t in fold_ticks]
    CALIBRATION.write_text(json.dumps(cal, indent=2))
    log("wrote", CALIBRATION)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry", action="store_true", help="find the stops but write nothing")
    args = p.parse_args(argv)

    arm = SO101(log=log)
    bus = arm.bus
    # The calibration the arm arrived with. Written once and never overwritten,
    # so a failed run cannot replace the original with its own half-state.
    backup_path = CALIBRATION.parent / "calibration_backup.json"
    if not backup_path.exists():
        backup_path.write_text(json.dumps(
            {JOINTS[k]: {"offset": ft.decode_sm(bus.read_retry(i, ft.HOMING_OFFSET), 11),
                         "min": bus.read_retry(i, ft.MIN_LIMIT), "max": bus.read_retry(i, ft.MAX_LIMIT),
                         "p_gain": bus.read_retry(i, ft.P_GAIN)}
             for k, i in enumerate(IDS)}, indent=2))
    backup = json.loads(backup_path.read_text())
    log("original EEPROM:", backup)

    rest_ticks = arm.ticks()
    rest = arm.to_rad(rest_ticks)
    log("rest pose (old calibration, rad):", np.round(rest, 3))
    sim_half = arm.check.upper / RAD_PER_TICK  # ticks from centre to sim limit

    # Open the angle limits so the goal can reach past the old ones.
    for i in IDS[:4]:
        bus.write(i, ft.LOCK, 0)
        bus.write(i, ft.MIN_LIMIT, 0)
        bus.write(i, ft.MAX_LIMIT, 4095)
    arm.lo[:4], arm.hi[:4] = 0, 4095

    fold_ticks = [arm.center + w / RAD_PER_TICK * arm.sign for w in arm.fold_path]  # frame-independent
    arm.configure(acceleration=40, torque_limit=600)
    arm.torque(True)
    stops = {}
    try:
        # Out of the fold, upward, before any sweep.
        arm.unfold()
        for j, stage_lo, stage_hi, tl in SWEEPS:
            q_lo, q_hi = ([rest[4] if v is None else v for v in st] for st in (stage_lo, stage_hi))
            centre = arm.center[j]
            bound = 1.20 * sim_half[j]
            log(f"{JOINTS[j]}: staging at {np.round(q_lo, 2)}")
            arm.move_to(q_lo, speed=0.25)
            lo = seek(arm, j, -1, centre - bound, tl)
            arm.move_to(q_lo, speed=0.25)
            log(f"{JOINTS[j]}: staging at {np.round(q_hi, 2)}")
            arm.move_to(q_hi, speed=0.25)
            hi = seek(arm, j, +1, centre + bound, tl)
            arm.move_to(q_hi, speed=0.25)
            width = (hi - lo) * RAD_PER_TICK
            sim_width = 2 * arm.check.upper[j]
            log(f"  {JOINTS[j]}: {lo:.0f}..{hi:.0f} ticks = {np.degrees(width):.1f} deg "
                f"(sim {np.degrees(sim_width):.1f}), midpoint {(lo + hi) / 2:.0f}")
            if width < TOO_SHORT * sim_width:
                raise RuntimeError(f"{JOINTS[j]} range is {width / sim_width:.0%} of the sim's - "
                                   "something other than the stop ended a sweep")
            stops[j] = (lo, hi)
            # Re-centre this joint immediately, so later staging poses are in
            # the new frame (software only; EEPROM is written at the end).
            arm.center[j] = (lo + hi) / 2
        log("sweeps done; back to rest")
        # The fold, re-expressed in the new frame of every re-centred joint.
        arm.fold_path = np.array([arm.to_rad(t) for t in fold_ticks])
        arm.move_to([0, 0, 0, 0, rest[4]], speed=0.25)
        arm.fold()
    except BaseException:
        arm.hold()
        for k, i in enumerate(IDS[:4]):
            bus.write(i, ft.MIN_LIMIT, backup[JOINTS[k]]["min"])
            bus.write(i, ft.MAX_LIMIT, backup[JOINTS[k]]["max"])
            bus.write(i, ft.LOCK, 1)
        log("ABORTED - holding position, old limits restored. Run `python -m hw.park` to bring it home.")
        raise

    result = {JOINTS[j]: {"stop_lo": lo, "stop_hi": hi} for j, (lo, hi) in stops.items()}
    (CALIBRATION.parent / "rehome_result.json").write_text(json.dumps(result, indent=2))

    if args.dry:
        log("dry run: nothing written. Restoring old limits.")
        arm.torque(False)
        for k, i in enumerate(IDS[:4]):
            bus.write(i, ft.MIN_LIMIT, backup[JOINTS[k]]["min"])
            bus.write(i, ft.MAX_LIMIT, backup[JOINTS[k]]["max"])
            bus.write(i, ft.LOCK, 1)
        return 0

    arm.torque(False)
    time.sleep(0.5)
    write_calibration(arm, backup, stops, fold_ticks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
