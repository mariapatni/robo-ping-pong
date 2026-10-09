"""Find a joint configuration that puts the paddle face somewhere, pointing
somewhere - the kinematics the oracle controller needs.

There is no analytic inverse kinematics here on purpose. The arm has five
joints and we are asking for a 3D position plus a face direction, which is six
constraints: the problem is over-determined and usually has no exact solution,
so what we actually want is the configuration that gets closest, with position
and orientation traded off explicitly. A coarse grid search followed by a local
refinement does that honestly, is a few lines, and cannot fail to converge.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .arm import ARM_JOINTS


@dataclass(frozen=True)
class PoseSolution:
    qpos: np.ndarray
    position: np.ndarray
    normal: np.ndarray
    position_error: float  # metres
    normal_error_deg: float


def _fk(model, data, qpos_adr, site, q):
    data.qpos[qpos_adr] = q
    mujoco.mj_kinematics(model, data)
    return data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3)[:, 2].copy()


def solve_face_pose(
    arm,
    target_pos,
    target_normal=None,
    site: int | None = None,
    orientation_weight: float = 0.08,
    coarse: tuple[int, int, int, int, int] = (17, 13, 13, 11, 9),
    refine_steps: int = 220,
    seed: int = 0,
) -> PoseSolution:
    """Closest configuration putting the paddle face at `target_pos`.

    `orientation_weight` is in metres per radian: it says how many metres of
    position error one radian of face-angle error is worth. The default makes
    10 degrees of tilt cost about 14 mm, which is the right order given a 40 mm
    ball and ~15 mm of real positioning error.
    """
    site = arm.tip_site if site is None else site
    target_pos = np.asarray(target_pos, float)
    want_normal = None if target_normal is None else np.asarray(target_normal, float)
    if want_normal is not None:
        want_normal = want_normal / np.linalg.norm(want_normal)

    data = mujoco.MjData(arm.model)

    def cost_of(q):
        pos, nrm = _fk(arm.model, data, arm.qpos_adr, site, q)
        c = float(np.linalg.norm(pos - target_pos))
        if want_normal is not None:
            c += orientation_weight * float(np.arccos(np.clip(nrm @ want_normal, -1.0, 1.0)))
        return c

    grids = [np.linspace(lo, hi, n) for lo, hi, n in zip(arm.lower, arm.upper, coarse)]
    mesh = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, len(ARM_JOINTS))
    best_q, best_c = mesh[0], np.inf
    for q in mesh:
        c = cost_of(q)
        if c < best_c:
            best_q, best_c = q.copy(), c

    # Local refinement: shrinking random walk. The coarse grid lands within a
    # cell of the optimum, so this only has to polish.
    rng = np.random.default_rng(seed)
    span = (arm.upper - arm.lower) / np.array(coarse, float)
    for i in range(refine_steps):
        scale = 0.6 ** (i / 40.0)
        q = np.clip(best_q + rng.normal(0.0, span * scale), arm.lower, arm.upper)
        c = cost_of(q)
        if c < best_c:
            best_q, best_c = q, c

    pos, nrm = _fk(arm.model, data, arm.qpos_adr, site, best_q)
    ang = 0.0 if want_normal is None else float(
        np.degrees(np.arccos(np.clip(nrm @ want_normal, -1.0, 1.0)))
    )
    return PoseSolution(
        qpos=best_q,
        position=pos,
        normal=nrm,
        position_error=float(np.linalg.norm(pos - target_pos)),
        normal_error_deg=ang,
    )
