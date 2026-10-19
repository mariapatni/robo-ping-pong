"""Lift the arm out of its rest fold to a raised pose, torque on.

    .venv/bin/python -m hw.unfold
"""

from __future__ import annotations

import numpy as np

from .so101 import SO101


def main() -> int:
    arm = SO101()
    arm.configure(acceleration=40)
    arm.torque(True)
    arm.unfold()
    print("raised at", np.round(arm.q(), 3), "load", np.round(arm.loads(), 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
