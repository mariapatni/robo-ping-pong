"""Where each arm can actually hit, and therefore where it is allowed to aim.

Three zones, and the distinction between them is the whole point:

  REACH      - cells the paddle face can occupy at all.
  UNDERHAND  - the subset it can occupy with the face open upward, i.e. the
               cells it can legally strike from.
  TARGET     - where a return is allowed to land: past the net, on the table,
               AND inside the OPPONENT's underhand zone. A ball that lands in
               bounds but outside that zone is a point won, not a rally - and a
               rally is what we are training.

Keeping the raw swept points rather than a binned grid means the arm's base can
be moved and the zones re-binned without re-running the kinematics, which is
what makes the placement sweep cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .arm import ARM_JOINTS
from .config import TableConfig
from .reach import STRIKE_BAND, UNDERHAND_MAX_TILT_DEG


@dataclass(frozen=True)
class SweptReach:
    """Raw paddle-face poses from a joint sweep, in the arm's own base frame."""

    pos: np.ndarray  # (n, 3)
    underhand: np.ndarray  # (n,) bool

    # No shifting here any more. The arm base IS the world origin and the table
    # carries its own `arm_inset`, so these poses are already in table
    # coordinates. Comparing two arm placements means comparing two
    # TableConfigs, not translating the points.


def sweep_face(
    arm,
    site: int | None = None,
    samples: tuple[int, int, int, int, int] = (29, 15, 15, 13, 11),
    band: tuple[float, float] = STRIKE_BAND,
    max_tilt_deg: float = UNDERHAND_MAX_TILT_DEG,
    blade_radius: float = 0.055,
    table_margin: float = 0.005,
) -> SweptReach:
    """Sweep joint space once, keeping every face pose inside the strike band.

    A pose also has to keep the BLADE off the table, not just the face centre.
    The rim of a disc tilted t off horizontal hangs R*sin(t) below its centre,
    so a near-vertical 55 mm blade reaches 55 mm down - and the strike band
    starts at 30 mm. Without this check half of all swings drove the paddle
    through the table, the worst by 100 mm, which on hardware is a broken
    paddle and in simulation is contact forces fighting the arm mid-stroke.
    """
    site = arm.tip_site if site is None else site
    grids = [np.linspace(lo, hi, n) for lo, hi, n in zip(arm.lower, arm.upper, samples)]
    mesh = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, len(ARM_JOINTS))

    data = mujoco.MjData(arm.model)
    pos = np.empty((len(mesh), 3))
    nrm = np.empty((len(mesh), 3))
    for i, q in enumerate(mesh):
        data.qpos[arm.qpos_adr] = q
        mujoco.mj_kinematics(arm.model, data)
        pos[i] = data.site_xpos[site]
        nrm[i] = data.site_xmat[site].reshape(3, 3)[:, 2]

    in_band = (pos[:, 2] >= band[0]) & (pos[:, 2] <= band[1])
    # Lowest point of the rim: centre minus R*sqrt(1 - nz^2).
    rim = pos[:, 2] - blade_radius * np.sqrt(np.clip(1.0 - nrm[:, 2] ** 2, 0.0, 1.0))
    clears = rim >= table_margin
    cos_limit = np.cos(np.radians(max_tilt_deg))
    underhand = in_band & clears & (nrm[:, 2] >= cos_limit) & (nrm[:, 0] > 0.0)
    return SweptReach(pos=pos[in_band], underhand=underhand[in_band])


def grid(table: TableConfig, cell: float = 0.02):
    nx = int(round(table.length / cell))
    ny = int(round(table.width / cell))
    return (
        np.linspace(table.x_near, table.x_far, nx + 1),
        np.linspace(-table.half_width, table.half_width, ny + 1),
    )


def occupancy(points: np.ndarray, table: TableConfig, cell: float = 0.02) -> np.ndarray:
    """Bin face positions onto the WHOLE table, not just one half."""
    xe, ye = grid(table, cell)
    occ = np.zeros((len(xe) - 1, len(ye) - 1), bool)
    keep = (
        (points[:, 0] >= table.x_near) & (points[:, 0] <= table.x_far)
        & (np.abs(points[:, 1]) <= table.half_width)
    )
    p = points[keep]
    if len(p):
        ix = np.clip(np.digitize(p[:, 0], xe) - 1, 0, occ.shape[0] - 1)
        iy = np.clip(np.digitize(p[:, 1], ye) - 1, 0, occ.shape[1] - 1)
        occ[ix, iy] = True
    return occ


def mirror(occ: np.ndarray) -> np.ndarray:
    """The same zone for the arm at the other end: a 180 degree turn about the
    table centre, which is a flip in BOTH axes. Not a mirror - a reflection
    would invert the joint chain's handedness and the zone would be wrong."""
    return occ[::-1, ::-1]


def zones(arm, table: TableConfig, cell: float = 0.02, swept=None):
    """Reach, underhand and target occupancy for arm A, plus arm B's underhand.

    Where the arm sits on the table is `table.arm_inset`; arm B is the same
    zone rotated 180 degrees about the table centre.
    """
    swept = swept or sweep_face(arm)
    pts = swept.pos
    reach = occupancy(pts, table, cell)
    under = occupancy(pts[swept.underhand], table, cell)
    b_under = mirror(under)

    xe, _ = grid(table, cell)
    past_net = (0.5 * (xe[:-1] + xe[1:]) > table.x_net)[:, None]
    target = b_under & past_net
    return {"reach": reach, "underhand": under, "b_underhand": b_under, "target": target}
