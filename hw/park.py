"""Fold the arm onto its rest and switch the torque off.

    .venv/bin/python -m hw.park            # fold up and relax
    .venv/bin/python -m hw.park --hold     # fold up, keep torque on
"""

from __future__ import annotations

import argparse

import numpy as np

from .so101 import SO101


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hold", action="store_true")
    args = p.parse_args(argv)
    arm = SO101()
    arm.configure(acceleration=40)
    arm.torque(True)
    arm.fold()
    if not args.hold:
        arm.torque(False)
    print("parked at", np.round(arm.q(), 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
