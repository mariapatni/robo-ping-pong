"""Feeds aimed at a chosen point, so training covers the whole strike zone.

The scripted launcher in sim/ball.py samples a spawn and a velocity and lets
the ball go where it goes - which bunches feeds down the middle, because that
is where a roughly-forward launch ends up. A policy trained on that learns to
cover the middle and nothing else.

This module inverts the problem: name the point you want the ball to arrive at,
and solve for the launch that gets it there. Sampling those points uniformly
over the arm's underhand zone is what makes the training distribution cover the
corners, the wide balls and the short ones.

The solve is a two-parameter shoot over (speed, elevation) with a lightweight
drag integrator rather than MuJoCo - it runs thousands of times during episode
setup, and only has to be accurate to a few millimetres against a 40 mm ball.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.81


@dataclass(frozen=True)
class Feed:
    pos: np.ndarray
    vel: np.ndarray
    target: np.ndarray
    miss: float  # metres between the simulated arrival and the target
    flight: float
    arrival_speed: float = 0.0  # ball speed at the target, m/s


def _fly(pos, vel, drag_k, mass, radius, restitution, dt=0.002, tmax=2.5):
    """Integrate one feed, bouncing once off the table.

    The bounce is not a detail - it is what makes a slow arrival possible. A
    ball that flies straight to the strike band has fallen from release height
    and arrives at ~2.9 m/s on gravity alone, far above the 1.0-1.9 m/s a rally
    runs at. After a bounce it is near the top of a much smaller arc, which is
    both slower and exactly how a real return arrives.
    """
    p = np.array(pos, float)
    v = np.array(vel, float)
    path = [p.copy()]
    bounce_at = None
    for _ in range(int(tmax / dt)):
        a = np.array([0.0, 0.0, -GRAVITY]) - (drag_k / mass) * np.linalg.norm(v) * v
        v = v + a * dt
        p = p + v * dt
        if p[2] <= radius and v[2] < 0:
            if bounce_at is None:
                p[2] = radius
                v[2] = -restitution * v[2]
                bounce_at = len(path)
            else:
                path.append(p.copy())
                break
        path.append(p.copy())
    return np.array(path), dt, bounce_at


def _closest(path, target, after=None):
    """Closest approach to the target, optionally only after the bounce."""
    start = 0 if after is None else after
    seg = path[start:]
    if len(seg) < 2:
        return float("inf"), start
    d = np.linalg.norm(seg - target, axis=1)
    i = int(np.argmin(d))
    return float(d[i]), start + i


def aim_feed(
    target,
    ball,
    spawn_x,
    table_e: float = 0.88,
    lateral_frac=(0.0, 0.75),
    spawn_z=(0.18, 0.55),
    arrival=(1.0, 1.9),
    speeds=(0.8, 4.0),
    elevations=(-20.0, 70.0),
    coarse: int = 15,
    refine: int = 120,
    seed: int = 0,
) -> Feed:
    """Solve for a launch from `spawn_x` that passes through `target`.

    The spawn's lateral position is placed on the line through the target, so
    the azimuth is fixed by geometry and only speed and elevation are searched.
    """
    target = np.asarray(target, float)
    rng = np.random.default_rng(seed)

    # Randomise WHERE the ball is shot from, while the target keeps the arrival
    # guaranteed. Varying the spawn matters because a fixed release point makes
    # incoming direction a deterministic function of the strike point, and a
    # policy can then infer where the ball is going from where it starts -
    # a shortcut that evaporates when the opponent is a second arm standing
    # somewhere else. Sampling the release breaks that correlation.
    if not np.isscalar(spawn_x):
        spawn_x = float(rng.uniform(*spawn_x))
    lat = float(rng.uniform(*lateral_frac))

    # The spawn sits "behind" the target along the direction of play, offset
    # laterally so the ball arrives travelling across the table as well as down
    # it. Its height is part of the search: a higher release buys fall time,
    # which is how a short ball gets placed slowly instead of quickly.
    def evaluate(speed, elev_deg, sz):
        """Miss distance, plus a penalty for arriving outside the rally band.

        Aiming on miss alone produces feeds launched at 3-4 m/s, because a flat
        fast ball is the easiest way to cross the table before falling into the
        strike band. Those are not balls the opponent arm could ever hit back -
        the sustainable rally window is 1.55-1.85 m/s - so a policy trained on
        them learns the wrong regime, with a third of the reaction time it will
        actually get. Penalising arrival speed buys the realistic solution,
        which is a higher, slower lob.
        """
        here = np.array([spawn_x, target[1] * lat, sz])
        to_t = target - here
        az = np.arctan2(to_t[1], to_t[0])
        e = np.radians(elev_deg)
        horiz = speed * np.cos(e)
        v0 = np.array([horiz * np.cos(az), horiz * np.sin(az), speed * np.sin(e)])
        path, dt, bounce = _fly(here, v0, ball.drag_k, ball.mass, ball.radius, table_e)
        if bounce is None:
            return 1e3, 1e3, 0.0, here, v0, 0.0
        d, i = _closest(path, target, after=bounce)
        j = max(1, min(i, len(path) - 1))
        v_arr = float(np.linalg.norm(path[j] - path[j - 1]) / dt)
        over = max(0.0, v_arr - arrival[1]) + max(0.0, arrival[0] - v_arr)
        return d + 0.25 * over, d, i * dt, here, v0, v_arr

    heights = np.linspace(spawn_z[0], spawn_z[1], 5)
    best = None
    for sz in heights:
        for sp in np.linspace(*speeds, coarse):
            for el in np.linspace(*elevations, coarse):
                c = evaluate(sp, el, sz)
                if best is None or c[0] < best[0][0]:
                    best = (c, sp, el, sz)

    (cost, d, t, here, v0, v_arr), sp, el, sz = best
    span_s = (speeds[1] - speeds[0]) / coarse
    span_e = (elevations[1] - elevations[0]) / coarse
    span_z = (spawn_z[1] - spawn_z[0]) / 5
    for i in range(refine):
        scale = 0.65 ** (i / 18.0)
        cs = float(np.clip(sp + rng.normal(0, span_s * scale), *speeds))
        ce = float(np.clip(el + rng.normal(0, span_e * scale), *elevations))
        cz = float(np.clip(sz + rng.normal(0, span_z * scale), *spawn_z))
        c = evaluate(cs, ce, cz)
        if c[0] < cost:
            (cost, d, t, here, v0, v_arr), sp, el, sz = c, cs, ce, cz

    return Feed(pos=here, vel=v0, target=target, miss=float(d), flight=float(t),
                arrival_speed=float(v_arr))


def sample_targets(occupancy, x_edges, y_edges, n: int, seed: int = 0, z=(0.06, 0.16)):
    """Uniform strike points over the cells the arm can legally hit.

    Uniform over the ZONE, not over the table: sampling the table and throwing
    away the misses would still bunch feeds wherever the zone is widest.
    """
    rng = np.random.default_rng(seed)
    cells = np.argwhere(occupancy)
    if not len(cells):
        raise ValueError("the underhand zone is empty")
    picks = cells[rng.integers(0, len(cells), n)]
    xs = rng.uniform(x_edges[picks[:, 0]], x_edges[picks[:, 0] + 1])
    ys = rng.uniform(y_edges[picks[:, 1]], y_edges[picks[:, 1] + 1])
    zs = rng.uniform(z[0], z[1], n)
    return np.stack([xs, ys, zs], axis=1)


def preroll_in_sim(world, feed, park_qpos=None, horizon: float = 0.6):
    """Fly the feed in MuJoCo with the arm out of the way, and report the truth.

    `aim_feed` solves against the lightweight integrator in this module, which
    is what makes the search fast enough to run per episode. That integrator is
    close but not identical to MuJoCo: measured over 24 feeds it puts the ball
    11.7 mm from the intended strike point and, more importantly, **17 ms early**
    - a systematic bias, not noise.

    17 ms at 1.83 m/s is 30 mm of ball travel, and the blade's usable radius is
    35 mm, so a swing scheduled off the analytic flight time arrives with almost
    no margin and misses about half the time. Planning against the real thing
    removes both errors at once.

    Returns (strike point, time of arrival, incoming velocity) as MuJoCo has
    them, which is what the swing should be built around.
    """
    import mujoco

    from .ball import drag_force

    model, data = world.model, world.data
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball")
    qadr, vadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

    # Swing the arm fully aside. Folded down it stands in the feed's path, and
    # the "arrival" found is then a rebound off the forearm.
    park = np.array([world.upper[0], 0.0, 0.0, 0.0, 0.0]) if park_qpos is None else np.asarray(park_qpos)

    mujoco.mj_resetData(model, data)
    data.qpos[world.qpos_adr] = park
    data.ctrl[world.actuator_ids] = park
    data.qpos[qadr:qadr + 3] = feed.pos
    data.qpos[qadr + 3:qadr + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[vadr:vadr + 6] = 0.0
    data.qvel[vadr:vadr + 3] = feed.vel
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    target = np.asarray(feed.target, float)
    best = (np.inf, target.copy(), 0.0, np.zeros(3))
    for k in range(int((feed.flight + horizon) / dt)):
        vel = data.qvel[vadr:vadr + 3].copy()
        data.xfrc_applied[body, :3] = drag_force(vel, world.cfg.ball.drag_k)
        mujoco.mj_step(model, data)
        pos = data.qpos[qadr:qadr + 3].copy()
        gap = float(np.linalg.norm(pos - target))
        if gap < best[0]:
            best = (gap, pos, k * dt, data.qvel[vadr:vadr + 3].copy())
    data.xfrc_applied[body, :3] = 0.0
    _, pos, t, vel = best
    return pos, float(t), vel


def correct_feed(world, feed, iters: int = 3, gain: float = 0.9):
    """Nudge a feed's launch velocity until MuJoCo puts the ball on the target.

    `aim_feed` solves against this module's own integrator, which is what makes
    it fast enough to run per episode. MuJoCo disagrees with it: measured over
    24 feeds the ball lands 11.7 mm from the intended strike point at the median
    and up to 112 mm at worst, and arrives a systematic 17 ms late.

    That is enough to decide an episode. The blade's usable radius is 35 mm, so
    a 112 mm error is a clean miss no matter how good the arm is - and one such
    episode was traced with the IK solving the strike pose to 0.0 mm and 0.0
    degrees, missing by 263 mm purely because the ball went elsewhere.

    Re-targeting the swing at where the ball actually goes is the wrong fix: it
    moves the strike point outside the reachable zone and makes contact worse
    (measured, 50% -> 35%). What is wanted is the ball on the point we chose,
    so this shoots: fly it, measure the miss, correct the launch, repeat. A
    couple of passes is enough because the error is near-linear in launch
    velocity over these distances.
    """
    target = np.asarray(feed.target, float)
    vel = np.array(feed.vel, float)
    best = feed
    for _ in range(iters):
        trial = Feed(pos=feed.pos, vel=vel, target=target, miss=feed.miss,
                     flight=feed.flight, arrival_speed=feed.arrival_speed)
        pos, t, v_arr = preroll_in_sim(world, trial)
        err = target - pos
        miss = float(np.linalg.norm(err))
        if miss < best.miss or best is feed:
            best = Feed(pos=feed.pos, vel=vel.copy(), target=target, miss=miss,
                        flight=t, arrival_speed=float(np.linalg.norm(v_arr)))
        if miss < 0.004:
            break
        # The ball is ballistic between launch and target, so to first order a
        # miss of `err` over a flight of `t` is cured by adding err/t to the
        # launch velocity. Under-relaxed because the bounce makes it nonlinear.
        vel = vel + gain * err / max(t, 0.1)
    return best
