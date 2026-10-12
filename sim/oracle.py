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
