"""How much of its own half can the arm actually reach?

This exists because the answer decides what the task can be. An SO-101 is about
40 cm from shoulder to gripper frame, and the half it is supposed to defend is
63 cm deep and 80 cm wide, so most of that half is simply not reachable and the
launcher ranges have to be aimed at the part that is.

The measurement is forward-kinematics only: sweep the joints on a grid, keep the
tip positions that land in a band above the table, and see which cells of the
half they cover. That is an optimistic bound - it ignores whether the paddle is
pointing anywhere useful and whether the arm could get there in time - so treat
it as a ceiling, not a promise.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .arm import ARM_JOINTS
from .config import TableConfig

# Height band above the table in which a paddle could meet the ball. The lower
# bound is a ball radius plus clearance; the upper is about where a slow feed
# still is when it crosses the near half.
STRIKE_BAND = (0.03, 0.22)


@dataclass(frozen=True)
class ReachMap:
    """Occupancy of the arm's half, on a grid, in the strike band."""

    occupied: np.ndarray  # bool, shape (nx, ny)
    x_edges: np.ndarray
    y_edges: np.ndarray
    max_x: float  # deepest reach into the half, metres from the base
    band: tuple[float, float]

    @property
    def fraction(self) -> float:
        return float(self.occupied.mean())

    @property
    def area(self) -> float:
        cell = (self.x_edges[1] - self.x_edges[0]) * (self.y_edges[1] - self.y_edges[0])
        return float(self.occupied.sum() * cell)


def reach_map(
    arm,
    table: TableConfig | None = None,
    band: tuple[float, float] = STRIKE_BAND,
    samples: tuple[int, int, int, int, int] = (49, 25, 25, 19, 3),
    cell: float = 0.02,
) -> ReachMap:
    """Sweep joint space and report which cells of the arm's half it covers.

    `samples` is the grid resolution per joint in ARM_JOINTS order. mj_kinematics
    rather than mj_forward: we only want site positions, and skipping the rest
    makes a half-million-pose sweep take a couple of seconds.
    """
    table = table or TableConfig()
    grids = [np.linspace(lo, hi, n) for lo, hi, n in zip(arm.lower, arm.upper, samples)]
    mesh = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, len(ARM_JOINTS))

    data = mujoco.MjData(arm.model)
    tips = np.empty((mesh.shape[0], 3))
    for i in range(mesh.shape[0]):
        data.qpos[arm.qpos_adr] = mesh[i]
        mujoco.mj_kinematics(arm.model, data)
        tips[i] = data.site_xpos[arm.tip_site]

    nx = max(1, int(round(table.half_length / cell)))
    ny = max(1, int(round(table.width / cell)))
    x_edges = np.linspace(0.0, table.half_length, nx + 1)
    y_edges = np.linspace(-table.half_width, table.half_width, ny + 1)

    keep = (
        (tips[:, 2] >= band[0])
        & (tips[:, 2] <= band[1])
        & (tips[:, 0] >= 0.0)
        & (tips[:, 0] <= table.half_length)
        & (np.abs(tips[:, 1]) <= table.half_width)
    )
    good = tips[keep]

    counts, _, _ = np.histogram2d(good[:, 0], good[:, 1], bins=(x_edges, y_edges))
    return ReachMap(
        occupied=counts > 0,
        x_edges=x_edges,
        y_edges=y_edges,
        max_x=float(good[:, 0].max()) if good.size else 0.0,
        band=band,
    )
