"""Step each joint and measure how the real servo answers.

    .venv/bin/python -m hw.joint_test                  # all joints, from the keep-up start pose
    .venv/bin/python -m hw.joint_test --joints elbow_flex --amps 0.1

From the pose the keep-up policy starts in, each joint in turn is stepped by
+a, back, -a, back, and its position sampled as fast as the bus allows. Per
step: time to 90% of the move, peak speed, overshoot, and where it settled.
These are the numbers the sim's actuator should be fitted to - today it is a
998 N.m/rad servo that reaches any target in a few milliseconds.

Out: out/hw/joint_test_<stamp>/{steps.npz, report.txt}.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from . import feetech as ft  # noqa: E402
from .so101 import IDS, JOINTS, RAD_PER_TICK, SO101, load_calibration  # noqa: E402


def step(arm: SO101, j: int, goal_tick: int, record_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Write one joint's goal and sample its position until record_s has passed."""
    sid = IDS[j]
    arm.bus.write(sid, ft.GOAL_POSITION, goal_tick)
    t0 = time.perf_counter()
    ts, ps = [], []
    while (now := time.perf_counter() - t0) < record_s:
        try:
            ps.append(arm.bus.read(sid, ft.POSITION))
            ts.append(now)
        except ft.BusError:
            pass
    return np.array(ts), np.array(ps, float)


def analyse(t, p, start, goal) -> dict:
    move = goal - start
    frac = (p - start) / move if move else np.zeros_like(p)
    i90 = np.argmax(frac >= 0.9) if np.any(frac >= 0.9) else None
    v = np.abs(np.diff(p) / np.maximum(np.diff(t), 1e-4)) * RAD_PER_TICK
    tail = p[t > t[-1] - 0.2]
    return {"t90_ms": None if i90 is None else 1000 * t[i90],
            "peak_v": float(v.max()) if len(v) else 0.0,
            "overshoot_pct": float(max(0.0, (frac.max() - 1) * 100)),
            "settled_err_deg": float(np.degrees((tail.mean() - goal) * RAD_PER_TICK)),
            "move_deg": float(np.degrees(move * RAD_PER_TICK))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--joints", nargs="*", default=list(JOINTS))
    p.add_argument("--amps", nargs="*", type=float, default=[0.1, 0.25], help="step sizes, rad")
    p.add_argument("--record", type=float, default=1.2, help="seconds sampled per step")
    p.add_argument("--torque-cap", type=float, default=0.6)
    p.add_argument("--uncalibrated", action="store_true",
                   help="allow before rehoming (responses are valid, the start pose only roughly right)")
    args = p.parse_args(argv)
    if load_calibration() is None and not args.uncalibrated:
        raise SystemExit("not rehomed yet; pass --uncalibrated to measure responses anyway")

    from sim.keepup import KeepUpEnv

    ready = KeepUpEnv().q0
    out = ROOT / "out" / "hw" / f"joint_test_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    arm = SO101()
    arm.configure(acceleration=0, torque_limit=int(args.torque_cap * 1000))
    arm.torque(True)
    rows, traces = [], {}
    try:
        arm.unfold()
        arm.move_to(ready, speed=0.25)
        for name in args.joints:
            j = JOINTS.index(name)
            home = int(arm.ticks()[j])
            for a in args.amps:
                for sgn in (+1, -1):
                    goal = int(np.clip(home + sgn * a / RAD_PER_TICK * arm.sign[j], arm.lo[j], arm.hi[j]))
                    if not arm.check.guard.is_safe(np.r_[arm.q()[:j], arm.to_rad(np.full(5, goal))[j], arm.q()[j + 1:]]):
                        print(f"{name} {sgn * a:+.2f}: skipped, too near the table in sim", flush=True)
                        continue
                    for label, frm, to in (("out", home, goal), ("back", goal, home)):
                        t, pos = step(arm, j, to, args.record)
                        r = analyse(t, pos, frm, to) | {"joint": name, "step": f"{sgn * a:+.2f} {label}"}
                        rows.append(r)
                        traces[f"{name}_{sgn * a:+.2f}_{label}"] = np.c_[t, pos]
                        print(f"{name:14s} {r['step']:>11s}  t90 {r['t90_ms'] or float('nan'):6.0f} ms  "
                              f"peak {r['peak_v']:5.2f} rad/s  overshoot {r['overshoot_pct']:4.1f}%  "
                              f"settled {r['settled_err_deg']:+5.1f} deg", flush=True)
                        time.sleep(0.2)
    finally:
        arm.hold()
        arm.configure(acceleration=40, torque_limit=int(args.torque_cap * 1000))
        arm.fold()
        arm.torque(False)

    np.savez(out / "steps.npz", **traces)
    lines = [f"{r['joint']:14s} {r['step']:>11s}  t90 {r['t90_ms'] or float('nan'):6.0f} ms  "
             f"peak {r['peak_v']:5.2f} rad/s  overshoot {r['overshoot_pct']:4.1f}%  "
             f"settled {r['settled_err_deg']:+5.1f} deg" for r in rows]
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
