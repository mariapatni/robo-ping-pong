"""The oracle: a scripted controller that cheats, to prove the task is winnable.

It is handed the ball's exact future instead of estimating it, and solves the
return in closed form. No learning. Its job is to answer "can this arm, with
this paddle, return this ball into the opponent's reachable zone, underhand?"
If the answer is no with perfect information, no policy will do better, and
that is worth knowing before spending a GPU on it.

The physics that makes it analytic: in a rigid-paddle collision only the
component of velocity along the paddle normal changes. So the velocity change

    dv = v_out - v_in

must itself point along the normal. That fixes the paddle orientation with no
search at all, and the required paddle speed follows from one scalar equation:

    |dv| = (1 + e) * (v_paddle . n - v_in . n)

Everything hard is then pushed into one 2D shoot for the outgoing velocity that
lands the ball where we want it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .feed import _closest, _fly

UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class ReturnPlan:
    normal: np.ndarray  # required paddle face normal
    paddle_speed: float  # along that normal, m/s
    v_out: np.ndarray  # resulting ball velocity
    landing: np.ndarray  # where it is predicted to land
    miss: float  # metres from the requested landing point
    underhand: bool
    tilt_deg: float  # face angle off vertical
    fouls_table: bool  # the blade rim would be under the table at this pose
    feasible: bool


def outgoing_for_landing(
    strike, landing, ball, table_e: float, speeds=(0.9, 3.2), elevations=(2.0, 62.0),
    coarse: int = 15, refine: int = 90, seed: int = 0,
):
    """Shoot for the launch velocity from `strike` that lands at `landing`."""
    strike = np.asarray(strike, float)
    landing = np.asarray(landing, float)
    rng = np.random.default_rng(seed)
    planar = landing[:2] - strike[:2]
    azimuth = np.arctan2(planar[1], planar[0])

    def vel(speed, elev_deg):
        e = np.radians(elev_deg)
        h = speed * np.cos(e)
        return np.array([h * np.cos(azimuth), h * np.sin(azimuth), speed * np.sin(e)])

    def miss_of(speed, elev_deg):
        v = vel(speed, elev_deg)
        path, _, bounce = _fly(strike, v, ball.drag_k, ball.mass, ball.radius, table_e)
        if bounce is None:
            return 1e3, v, strike
        p = path[bounce]
        return float(np.linalg.norm(p[:2] - landing[:2])), v, p

    best = None
    for sp in np.linspace(*speeds, coarse):
        for el in np.linspace(*elevations, coarse):
            d, v, p = miss_of(sp, el)
            if best is None or d < best[0]:
                best = (d, v, p, sp, el)
    d, v, p, sp, el = best
    ss = (speeds[1] - speeds[0]) / coarse
    se = (elevations[1] - elevations[0]) / coarse
    for i in range(refine):
        scale = 0.65 ** (i / 14.0)
        cs = float(np.clip(sp + rng.normal(0, ss * scale), *speeds))
        ce = float(np.clip(el + rng.normal(0, se * scale), *elevations))
        dd, vv, pp = miss_of(cs, ce)
        if dd < d:
            d, v, p, sp, el = dd, vv, pp, cs, ce
    return v, p, d


def paddle_for(v_in, v_out, paddle_e: float):
    """The face normal and paddle speed that turn `v_in` into `v_out`.

    Exact for a rigid paddle with no friction. Spin is not modelled, which is
    the same assumption the rest of v1 makes.
    """
    v_in = np.asarray(v_in, float)
    v_out = np.asarray(v_out, float)
    dv = v_out - v_in
    mag = float(np.linalg.norm(dv))
    if mag < 1e-9:
        return UP.copy(), 0.0
    n = dv / mag
    # |dv| = (1 + e) (v_p . n - v_in . n), with the paddle moving along n.
    speed = float(v_in @ n) + mag / (1.0 + paddle_e)
    return n, speed


def plan_return(
    strike, v_in, landing, ball, table_e: float, paddle_e: float,
    max_tilt_deg: float = 60.0, max_paddle_speed: float = 1.5, seed: int = 0,
    blade_radius: float = 0.055, table_margin: float = 0.005,
) -> ReturnPlan:
    """Plan one underhand return from `strike` to `landing`, and say whether
    the arm could actually do it."""
    v_out, predicted, miss = outgoing_for_landing(strike, landing, ball, table_e, seed=seed)
    n, speed = paddle_for(v_in, v_out, paddle_e)

    # The face may present either way round; pick the side that is open upward.
    if n[2] < 0:
        n, speed = -n, -speed
    tilt = float(np.degrees(np.arccos(np.clip(n @ UP, -1.0, 1.0))))
    underhand = tilt <= max_tilt_deg and n[0] > 0.0

    # The rim of a disc tilted t off horizontal hangs R*sin(t) below its
    # centre. A plan that demands a steep face low over the table is asking the
    # arm to swing through the table, which is a broken paddle on hardware.
    rim = float(strike[2] - blade_radius * np.sqrt(max(0.0, 1.0 - n[2] ** 2)))
    fouls = rim < table_margin

    feasible = bool(underhand and not fouls and 0.0 <= speed <= max_paddle_speed and miss < 0.08)

    return ReturnPlan(
        normal=n, paddle_speed=speed, v_out=v_out, landing=predicted,
        miss=float(miss), underhand=bool(underhand), tilt_deg=tilt,
        fouls_table=bool(fouls), feasible=feasible,
    )


def plan_within_reach(
    arm,
    guard,
    strike,
    v_in,
    landing,
    ball,
    table_e: float,
    paddle_e: float,
    max_tilt_deg: float = 60.0,
    max_paddle_speed: float = 1.5,
    n_normals: int = 48,
    seed: int = 0,
):
    """Plan the return from what the arm CAN do, not from what we wish it would.

    `plan_return` works the other way round: pick where the ball should land,
    derive the one face normal that sends it there, and hand that to the IK. It
    is the obvious decomposition and it is backwards, because nothing in it ever
    asks whether the arm can hold that face. Measured, 29% of the poses it
    demanded put the forearm inside the table, and the arm started its swing
    from inside the table because the episode set those joint angles directly.

    Here the achievable set comes first. Candidate face normals are sampled
    across the underhand cone, each is solved and tested against the guard, and
    only the ones the arm can actually hold are scored - by where the ball would
    land if struck that way. An intersecting pose is not penalised, it is simply
    not a candidate.

    Returns (plan, pose) with pose None when nothing in the cone is reachable
    without touching the table, which is a legitimate answer: that ball is not
    returnable and should be scored as a miss rather than attempted.
    """
    from .pose import solve_face_pose_ik

    strike = np.asarray(strike, float)
    v_in = np.asarray(v_in, float)
    landing = np.asarray(landing, float)
    rng = np.random.default_rng(seed)

    # The ideal normal, as the old planner would have demanded it. Candidates
    # are spread around it, so the search starts where the answer probably is.
    ideal = plan_return(strike, v_in, landing, ball, table_e, paddle_e,
                        max_tilt_deg=max_tilt_deg, seed=seed)

    cos_limit = np.cos(np.radians(max_tilt_deg))
    candidates = [ideal.normal]
    for _ in range(n_normals - 1):
        # Uniform in the spherical cap about vertical, then kept only if it
        # also points across the table.
        z = rng.uniform(cos_limit, 1.0)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        r = np.sqrt(max(0.0, 1.0 - z * z))
        n = np.array([r * np.cos(phi), r * np.sin(phi), z])
        if n[0] > 0.0:
            candidates.append(n)

    best = None
    for k, n in enumerate(candidates):
        sol = solve_face_pose_ik(arm, strike, n, seed=seed + k, guard=guard)
        if not guard.is_safe(sol.qpos):
            continue  # the arm cannot hold this face without touching the table
        if sol.position_error > 0.035:
            continue  # it cannot get the paddle there either

        # Score by what this ACHIEVABLE face would actually do to the ball.
        actual = sol.normal / np.linalg.norm(sol.normal)
        speed = float(np.clip(_speed_along(v_in, actual, paddle_e, landing, strike,
                                           ball, table_e), 0.0, max_paddle_speed))
        v_out = _bounce(v_in, actual, speed, paddle_e)
        land = _landing_of(strike, v_out, ball, table_e)
        if land is None:
            continue
        miss = float(np.linalg.norm(land[:2] - landing[:2]))
        tilt = float(np.degrees(np.arccos(np.clip(actual @ UP, -1.0, 1.0))))
        score = miss + 0.002 * sol.position_error * 1000
        if best is None or score < best[0]:
            best = (score, ReturnPlan(normal=actual, paddle_speed=speed, v_out=v_out,
                                      landing=land, miss=miss, underhand=True,
                                      tilt_deg=tilt, fouls_table=False,
                                      feasible=miss < 0.12), sol)
    if best is None:
        return None, None
    return best[1], best[2]


def _bounce(v_in, n, paddle_speed, e):
    """Outgoing ball velocity for a paddle of normal `n` moving at `paddle_speed`."""
    vn_in = float(v_in @ n)
    vn_out = -e * (vn_in - paddle_speed) + paddle_speed
    return v_in - vn_in * n + vn_out * n


def _speed_along(v_in, n, e, landing, strike, ball, table_e):
    """Paddle speed along `n` that best sends the ball toward `landing`.

    A short scalar search rather than a solve: the landing point is not a
    closed form of paddle speed once drag and the bounce are in the way.
    """
    best, best_miss = 0.0, np.inf
    for sp in np.linspace(0.0, 1.5, 16):
        land = _landing_of(strike, _bounce(v_in, n, sp, e), ball, table_e)
        if land is None:
            continue
        miss = float(np.linalg.norm(land[:2] - np.asarray(landing, float)[:2]))
        if miss < best_miss:
            best, best_miss = sp, miss
    return best


def _landing_of(strike, v_out, ball, table_e):
    """Where a ball leaving `strike` at `v_out` first touches down."""
    from .feed import _fly

    path, _, bounce = _fly(strike, v_out, ball.drag_k, ball.mass, ball.radius, table_e)
    if bounce is None:
        return None
    return path[bounce]
