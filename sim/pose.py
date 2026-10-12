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


def solve_face_pose_ik(
    arm,
    target_pos,
    target_normal=None,
    site: int | None = None,
    q0=None,
    iters: int = 320,
    damping: float = 0.25,
    normal_weight: float = 0.35,
    restarts: int = 6,
    seed: int = 0,
) -> PoseSolution:
    """Damped least-squares IK on the paddle face, position and normal together.

    Replaces the coarse grid plus random walk in `solve_face_pose`. That search
    was not wrong so much as too blunt: a 5-joint arm meeting a 3-position plus
    2-normal target is only just determined, so the solution is a thin sliver in
    joint space. Sampling finds its neighbourhood and then cannot refine into
    it - measured, the grid solver left 12.6 degrees of normal error at contact
    where a dense sweep showed poses within 1 to 8 degrees existed.

    That error matters more than it looks. A flat face reflects, so d degrees of
    tilt turns the outgoing ball by 2d; over 0.6 m of flight 12.6 degrees throws
    the landing point 340 mm, and the target zone is 150 mm deep. Position error
    is far more forgiving: the blade is 110 mm across, so tens of millimetres
    still strike cleanly. Hence `normal_weight` is in metres per radian and is
    deliberately large - orientation is the expensive axis.

    Jacobians come from `mj_jacSite`, so each step is a proper gradient rather
    than a guess, and several restarts cover the arm's elbow-up/elbow-down
    branches.
    """
    site = arm.tip_site if site is None else site
    target_pos = np.asarray(target_pos, float)
    want = None if target_normal is None else np.asarray(target_normal, float) / np.linalg.norm(target_normal)

    model = arm.model
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    dof = arm.qvel_adr

    def solve_from(q):
        q = np.clip(q.copy(), arm.lower, arm.upper)
        for it in range(iters):
            # Anneal the damping. Heavy damping early keeps the first steps
            # stable through the singularities a 5-DoF arm starts near; holding
            # it there afterwards is what leaves residual error, because damping
            # trades accuracy for stability and never reaches zero.
            lam = damping * (0.04 / damping) ** min(1.0, it / (0.6 * iters))
            data.qpos[arm.qpos_adr] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)

            pos = data.site_xpos[site]
            err = [target_pos - pos]
            mujoco.mj_jacSite(model, data, jacp, jacr, site)
            J = [jacp[:, dof]]

            if want is not None:
                nrm = data.site_xmat[site].reshape(3, 3)[:, 2]
                # The rotation that carries the current normal onto the wanted
                # one, as a rotation vector: axis = n x n_want, angle between.
                axis = np.cross(nrm, want)
                s = np.linalg.norm(axis)
                ang = np.arctan2(s, float(nrm @ want))
                rot_err = (axis / s * ang) if s > 1e-9 else np.zeros(3)
                err.append(normal_weight * rot_err)
                J.append(normal_weight * jacr[:, dof])

            e = np.concatenate(err)
            Jm = np.vstack(J)
            # Damped least squares: stable through the singularities a 5-DoF
            # arm sits on constantly, where a plain pseudo-inverse blows up.
            dq = Jm.T @ np.linalg.solve(Jm @ Jm.T + lam**2 * np.eye(len(e)), e)
            step = np.clip(dq, -0.25, 0.25)
            q = np.clip(q + step, arm.lower, arm.upper)
            if np.abs(step).max() < 1e-6:
                break
        return q

    best = None
    seeds = [np.zeros(5) if q0 is None else np.asarray(q0, float)]
    seeds += [rng.uniform(arm.lower, arm.upper) for _ in range(restarts)]
    for q_start in seeds:
        q = solve_from(q_start)
        data.qpos[arm.qpos_adr] = q
        mujoco.mj_kinematics(model, data)
        pos = data.site_xpos[site].copy()
        nrm = data.site_xmat[site].reshape(3, 3)[:, 2].copy()
        pe = float(np.linalg.norm(pos - target_pos))
        ae = 0.0 if want is None else float(np.degrees(np.arccos(np.clip(nrm @ want, -1.0, 1.0))))
        cost = pe + normal_weight * np.radians(ae)
        if best is None or cost < best[0]:
            best = (cost, q, pos, nrm, pe, ae)

    _, q, pos, nrm, pe, ae = best
    return PoseSolution(qpos=q, position=pos, normal=nrm, position_error=pe, normal_error_deg=ae)
