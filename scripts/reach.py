"""Print the arm's reachable footprint over its own half of the table.

    .venv/bin/python scripts/reach.py

The map is drawn looking down on the table: rows are depth away from the arm's
base, columns are across. This is the number that decides whether the task is
"return the ball" or "return the ball if it comes to you".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.env import PingPongEnv  # noqa: E402
from sim.reach import STRIKE_BAND, reach_map  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--low", type=float, default=STRIKE_BAND[0], help="bottom of the strike band, m above table")
    p.add_argument("--high", type=float, default=STRIKE_BAND[1], help="top of the strike band")
    p.add_argument("--cell", type=float, default=0.02, help="grid cell size, m")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    env = PingPongEnv()
    t = env.cfg.table
    rm = reach_map(env, t, band=(args.low, args.high), cell=args.cell)

    print(f"table        : {t.length * 1000:.0f} x {t.width * 1000:.0f} mm, top at z = 0")
    print(f"arm's half   : x in [0, {t.half_length:.3f}] m, y in +/-{t.half_width:.3f} m")
    print(f"strike band  : z in [{args.low:.3f}, {args.high:.3f}] m above the table")
    print(f"deepest reach: x = {rm.max_x:.3f} m ({rm.max_x / t.half_length * 100:.0f}% of the half's depth)")
    print(f"reachable    : {rm.area * 1e4:.0f} cm^2 of {t.half_length * t.width * 1e4:.0f} cm^2"
          f"  =  {rm.fraction * 100:.1f}% of the half\n")

    for ix in range(rm.occupied.shape[0] - 1, -1, -1):
        depth = 0.5 * (rm.x_edges[ix] + rm.x_edges[ix + 1])
        row = "".join("#" if c else "." for c in rm.occupied[ix])
        print(f"  x={depth:5.3f}  {row}")
    print(f"\n  {' ' * 10}y from {-t.half_width:+.2f} to {t.half_width:+.2f} m (base of the arm at x=0, y=0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
