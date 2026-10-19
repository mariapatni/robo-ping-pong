"""Move joints to absolute angles (rad), one call at a time, and report.

    .venv/bin/python -m hw.jog elbow_flex=0.6 wrist_flex=-0.3 [--speed 0.4] [--relax]

Only the named joints move; the others hold where they are. Torque is left on
unless --relax is given.
"""

from __future__ import annotations

import argparse

import numpy as np

from . import feetech as ft
from .so101 import IDS, JOINTS, SO101


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("moves", nargs="*", help="joint=rad")
    p.add_argument("--speed", type=float, default=0.4)
    p.add_argument("--relax", action="store_true")
    p.add_argument("--no-check", action="store_true", help="skip the sim table check")
    args = p.parse_args(argv)
    arm = SO101()
    arm.configure(acceleration=40)
    arm.torque(True)
    q = arm.q()
    for m in args.moves:
        name, v = m.split("=")
        q[JOINTS.index(name)] = float(v)
    try:
        reached = arm.move_to(q, speed=args.speed, check=not args.no_check)
    finally:
        loads = [ft.decode_sm(arm.bus.read_retry(i, ft.LOAD), 10) for i in IDS]
        print("q   ", np.round(arm.q(), 3))
        print("load", loads)
    if args.relax:
        arm.torque(False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
