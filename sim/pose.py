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
    blade_radius: float = 0.055,
    table_margin: float = 0.005,
    guard=None,
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

    # Clearance is a hard filter over restarts, not a term in the cost. The
    # swept pose set already excludes blades that foul the table, but the IK
    # searched freely and could land outside it - which is how 37% of swings
    # still drove the paddle under the surface after the zone was filtered.
    # A penalty would only trade clearance against accuracy; a filter cannot.
    best = None
    best_clear = None
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
        candidate = (cost, q, pos, nrm, pe, ae)
        if best is None or cost < best[0]:
            best = candidate
        # `guard` tests every arm geom against the table using MuJoCo's own
        # collision detection; the blade-only fallback is a weaker check kept
        # for callers that have no model handy. The difference matters: with
        # the blade test alone, 29% of returned strike poses still had the
        # FOREARM inside the table, because nothing was looking at it.
        ok = (guard.is_safe(q) if guard is not None
              else blade_lowest(pos, nrm, blade_radius) >= table_margin)
        if ok:
            if best_clear is None or cost < best_clear[0]:
                best_clear = candidate

    # Prefer a pose that clears; fall back to the best found only when no
    # restart cleared at all, so the caller still gets something to inspect.
    best = best_clear if best_clear is not None else best

    _, q, pos, nrm, pe, ae = best
    return PoseSolution(qpos=q, position=pos, normal=nrm, position_error=pe, normal_error_deg=ae)


def blade_lowest(face_pos, face_normal, blade_radius: float) -> float:
    """Height of the lowest point of the blade rim.

    The rim is a circle of radius R in the plane normal to the face, so its
    lowest point sits R*sqrt(1 - nz^2) below the centre: nothing below the
    centre when the face is horizontal, a full radius when it is vertical.

    This is the quantity a strike pose has to keep above the table. The face
    CENTRE clearing the table is not enough, and that was the bug: strike
    points were sampled down to z = 0.03 while a near-vertical 55 mm blade
    reaches 55 mm below its own centre.
    """
    n = np.asarray(face_normal, float)
    n = n / max(np.linalg.norm(n), 1e-9)
    return float(np.asarray(face_pos, float)[2] - blade_radius * np.sqrt(max(0.0, 1.0 - n[2] ** 2)))


def clears_table(face_pos, face_normal, blade_radius: float, margin: float = 0.005) -> bool:
    """Would a paddle at this pose stay out of the table?"""
    return blade_lowest(face_pos, face_normal, blade_radius) >= margin


def windup_pose(strike, face_normal, draw_back: float = 0.055, blade_radius: float = 0.055,
                margin: float = 0.008):
    """Where to start the swing: drawn back along the hitting direction.

    The wind-up used to be a blind 45 mm drop in z, which is the one direction
    guaranteed to make trouble - it aims the blade at the table. Measured, half
    of all swings put the rim below the table top and the worst reached 100 mm
    through it. On hardware that is a broken paddle, and in simulation it is
    contact forces fighting the arm through the whole stroke.

    Drawing back along the face normal instead is both safer and more useful:
    it is the direction the paddle will travel through the ball, so the stroke
    arrives with velocity pointing where the ball should go rather than having
    to turn a corner. If that still fouls the table, the draw is tilted up
    until it clears.
    """
    strike = np.asarray(strike, float)
    n = np.asarray(face_normal, float)
    n = n / max(np.linalg.norm(n), 1e-9)

    for lift in np.linspace(0.0, 1.0, 11):
        direction = n * (1.0 - lift) + np.array([0.0, 0.0, 1.0]) * lift
        direction /= max(np.linalg.norm(direction), 1e-9)
        candidate = strike - draw_back * direction
        if clears_table(candidate, n, blade_radius, margin):
            return candidate
    # Nothing clears: back off straight up, which always does.
    return strike + np.array([0.0, 0.0, draw_back])


def safe_windup(
    arm,
    strike,
    face_normal,
    hit_qpos,
    blade_radius: float = 0.055,
    draw_back: float = 0.055,
    margin: float = 0.008,
    site: int | None = None,
    samples: int = 14,
    seed: int = 0,
):
    """A wind-up pose whose whole swing keeps the blade off the table.

    Both endpoints clearing is not enough. The swing interpolates in JOINT
    space, and a straight line in joint space is a curve in Cartesian space, so
    the blade can dip between two perfectly legal poses. Choosing the endpoints
    alone left 14% of swings still scraping the table.

    So the path itself is checked: lift the draw-back direction until no sample
    along the interpolation puts the rim below `margin`.
    """
    site = arm.tip_site if site is None else site
    strike = np.asarray(strike, float)
    n = np.asarray(face_normal, float)
    n = n / max(np.linalg.norm(n), 1e-9)

    data = mujoco.MjData(arm.model)

    def path_clearance(wind_qpos):
        worst = np.inf
        for a in np.linspace(0.0, 1.0, samples):
            data.qpos[arm.qpos_adr] = wind_qpos + (hit_qpos - wind_qpos) * a
            mujoco.mj_kinematics(arm.model, data)
            pos = data.site_xpos[site]
            nrm = data.site_xmat[site].reshape(3, 3)[:, 2]
            worst = min(worst, blade_lowest(pos, nrm, blade_radius))
        return worst

    best = None
    for lift in np.linspace(0.0, 1.0, 9):
        direction = n * (1.0 - lift) + np.array([0.0, 0.0, 1.0]) * lift
        direction /= max(np.linalg.norm(direction), 1e-9)
        target = strike - draw_back * direction
        sol = solve_face_pose_ik(arm, target, n, site=site, seed=seed)
        clear = path_clearance(sol.qpos)
        if best is None or clear > best[0]:
            best = (clear, sol)
        if clear >= margin:
            return sol
    return best[1]


def swing_through(strike, face_normal, paddle_speed: float, swing_time: float = 0.34,
                  min_follow: float = 0.015, max_follow: float = 0.090):
    """Where the paddle should be heading, past the ball rather than at it.

    A swing that ends at the contact point is decelerating when it arrives.
    Measured: the paddle delivered 0.14 m/s where the plan asked for 0.70, and
    was at a third of its own peak speed at the moment it struck. Four fifths
    of the intended energy was simply not there, which is what puts 37% of
    returns back on the arm's own half.

    So the target is placed BEYOND the strike point along the hitting
    direction, and the swing is timed to pass THROUGH the strike point rather
    than stop on it - which is what a follow-through is for in any racket
    sport. You accelerate through the ball, not to it.

    The distance is proportional to the speed wanted, because not every stroke
    should follow through the same amount: a gentle block needs almost none,
    and giving it a long one overshoots the ball and puts the paddle somewhere
    useless for the next exchange. `min_follow` keeps a short stroke honest and
    `max_follow` keeps a hard one inside the arm's reach.

    Returns (through_point, fraction), where `fraction` is how far along the
    wind-up-to-through path the strike point sits. The caller times its ramp so
    that fraction of it elapses at contact.
    """
    strike = np.asarray(strike, float)
    n = np.asarray(face_normal, float)
    n = n / max(np.linalg.norm(n), 1e-9)

    # Travel beyond contact, scaled by how hard this stroke is meant to be.
    follow = float(np.clip(paddle_speed * swing_time * 0.6, min_follow, max_follow))
    return strike + follow * n, follow
