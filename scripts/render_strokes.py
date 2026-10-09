"""Show each kind of stroke and what the cost function pays for it.

    .venv/bin/python scripts/render_strokes.py      # -> out/strokes.mp4

Four scripted strokes against the same incoming feed, scored by sim/reward.py
from what actually happened in contact, not from what was intended. The point
is to make the reward legible: you should be able to watch it and see why the
underhand clip scores seven times the overhand one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.ball import BALL_BODY, drag_force  # noqa: E402
from sim.config import EnvConfig  # noqa: E402
from sim.env import PingPongEnv  # noqa: E402
from sim.paddle import PADDLE_BODY, PaddleConfig  # noqa: E402
from sim.pose import solve_face_pose  # noqa: E402
from sim.reward import exchange_reward  # noqa: E402

# Through PingPongEnv rather than a hand-rolled modify hook, so the 2 kHz
# timestep comes with it. At the model's default 200 Hz the 2.7 g ball/table
# contact injects energy and the ball leaves at 14 m - which is exactly the
# trap this script fell into first.
CFG = EnvConfig(paddle=PaddleConfig())
TABLE, BALL = CFG.table, CFG.ball
INTERCEPT = np.array([0.36, 0.0, 0.11])
AIM = (TABLE.length * 0.75, 0.0)


def unit(v):
    v = np.array(v, float)
    return v / np.linalg.norm(v)


# name, face normal at contact, how far the stroke lifts, sweep seconds
# A slower sweep is a gentler paddle: the whole stroke travels the same
# distance, so doubling the duration halves the speed at contact. Sustaining a
# rally needs only ~0.25 m/s, which is why these are so leisurely.
# name, face normal at contact, lift distance, sweep seconds, position offset.
# The rim clip is the interesting one: its face angle is perfectly good and it
# fails purely on position, which is the failure a real arm with ~15 mm of
# repeatability error will actually make.
STROKES = (
    ("Underhand lift", unit([0.35, 0.0, 0.94]), 0.050, 0.55, (0.0, 0.0, 0.0)),
    ("Overhand chop", unit([0.35, 0.0, -0.94]), -0.050, 0.55, (0.0, 0.0, 0.0)),
    ("Underhand, too hard", unit([0.75, 0.0, 0.66]), 0.075, 0.13, (0.0, 0.0, 0.0)),
    ("Rim clip (65 mm off-centre)", unit([0.35, 0.0, 0.94]), 0.050, 0.55, (0.0, 0.065, 0.0)),
)


def build(offscreen):
    world = PingPongEnv(cfg=CFG, offscreen=offscreen)
    world.tip_site = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_SITE, "paddle_face")
    return world


def paddle_geoms(model) -> set[int]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PADDLE_BODY)
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] == bid}


def camera():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.45, 0.0, 0.10)
    cam.distance = 1.25
    cam.azimuth = 118.0
    cam.elevation = -14.0
    return cam


def feed_velocity(start, target, flight):
    """Velocity to get from `start` to `target` in `flight` seconds, ballistic."""
    d = np.asarray(target, float) - np.asarray(start, float)
    v = d / flight
    v[2] += 0.5 * 9.81 * flight
    return v


def preroll(arm, start, vel, qadr, vadr, ball_b):
    """Roll the ball with no arm in the way and find where it actually goes.

    The intercept cannot be imposed: drag means a ballistic aim point is not
    where the ball ends up, and a paddle parked at the wrong place simply
    misses. So the ball defines the intercept, and the stroke is built around
    it - which is exactly what the oracle controller will do.
    """
    model, data = arm.model, arm.data
    mujoco.mj_resetData(model, data)
    # Park the arm swung fully aside. An earlier version folded it downward,
    # which left it standing in the feed's path: the ball struck the forearm
    # and the "intercept" the preroll found was a rebound off the arm.
    park = np.array([arm.upper[0], 0.0, 0.0, 0.0, 0.0])
    data.qpos[arm.qpos_adr] = park
    data.ctrl[arm.actuator_ids] = park
    data.qpos[qadr:qadr + 3] = start
    data.qpos[qadr + 3:qadr + 7] = (1, 0, 0, 0)
    data.qvel[vadr:vadr + 6] = 0.0
    data.qvel[vadr:vadr + 3] = vel
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    best = None
    for k in range(int(1.2 / dt)):
        data.xfrc_applied[ball_b, :3] = drag_force(data.qvel[vadr:vadr + 3], BALL.drag_k)
        mujoco.mj_step(model, data)
        p = data.qpos[qadr:qadr + 3].copy()
        if 0.05 <= p[2] <= 0.20 and 0.20 <= p[0] <= 0.45:
            best = (k * dt, p)
            break
    data.xfrc_applied[ball_b, :3] = 0.0
    return best


def run_stroke(arm, name, normal, lift_m, sweep, offset, fps, frames, label_rows):
    model, data = arm.model, arm.data
    pads = paddle_geoms(model)
    ball_g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    ball_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BALL_BODY)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, BALL_BODY)
    qadr, vadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])

    flight = 0.55
    start = np.array([0.95, 0.0, 0.22])
    vel = feed_velocity(start, INTERCEPT, flight)

    found = preroll(arm, start, vel, qadr, vadr, ball_b)
    if found is None:
        raise RuntimeError(f"{name}: the feed never entered the strike band")
    flight, intercept = found

    aim_at = intercept + np.asarray(offset, float)
    hit = solve_face_pose(arm, aim_at, normal)
    pre = solve_face_pose(arm, aim_at - np.array([0.0, 0.0, lift_m]), normal)

    mujoco.mj_resetData(model, data)
    data.qpos[arm.qpos_adr] = pre.qpos
    arm.command(pre.qpos)

    data.qpos[qadr:qadr + 3] = start
    data.qpos[qadr + 3:qadr + 7] = (1, 0, 0, 0)
    data.qvel[vadr:vadr + 6] = 0.0
    data.qvel[vadr:vadr + 3] = vel
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    steps_per_frame = max(1, round(1.0 / (fps * dt)))
    total = int(1.5 / dt)
    sweep_start = flight - sweep

    contact_n = face_n = vel_out = None
    landing = None
    prev_z = 1.0

    for k in range(total):
        t = k * dt
        if t < sweep_start:
            arm.command(pre.qpos)
        else:
            a = min(1.0, (t - sweep_start) / sweep)
            arm.command(pre.qpos + (hit.qpos - pre.qpos) * a)

        data.xfrc_applied[ball_b, :3] = drag_force(data.qvel[vadr:vadr + 3], BALL.drag_k)
        mujoco.mj_step(model, data)

        if contact_n is None:
            for c in range(data.ncon):
                con = data.contact[c]
                pair = {con.geom1, con.geom2}
                if ball_g in pair and pair & pads:
                    contact_n = con.frame[:3].copy()
                    face_n = data.site_xmat[arm.tip_site].reshape(3, 3)[:, 2].copy()
                    break
        elif vel_out is None and t > 0.02 + flight:
            vel_out = data.qvel[vadr:vadr + 3].copy()

        # Landing: the first time the ball drops back to the table plane after
        # contact. Taken at the plane crossing rather than at a bounce, so a
        # ball that sails past the table edge is still scored where it crossed.
        z = data.qpos[qadr + 2]
        if contact_n is not None and landing is None and prev_z > BALL.radius >= z:
            landing = data.qpos[qadr:qadr + 2].copy()
        prev_z = z

        if k % steps_per_frame == 0:
            frames.append((data.time, name))
            yield

    data.xfrc_applied[ball_b, :3] = 0.0
    r = exchange_reward(
        contact_normal=contact_n, face_normal=face_n,
        ball_vel_out=vel_out if vel_out is not None else np.zeros(3),
        landing_xy=landing, table=TABLE, aim_xy=AIM,
    )
    label_rows.append((name, r, landing))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("out/strokes.mp4"))
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1100)
    p.add_argument("--height", type=int, default=680)
    args = p.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    arm = build((args.width, args.height))
    cam = camera()
    rows, images = [], []

    with mujoco.Renderer(arm.model, height=args.height, width=args.width) as r:
        for name, normal, lift_m, sweep, offset in STROKES:
            frames = []
            for _ in run_stroke(arm, name, normal, lift_m, sweep, offset, args.fps, frames, rows):
                r.update_scene(arm.data, camera=cam)
                images.append((r.render(), name))

    by_name = {n: (rw, land) for n, rw, land in rows}
    out = []
    for img, name in images:
        f = np.ascontiguousarray(img)
        rw, land = by_name[name]
        cv2.putText(f, name, (26, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        verdict = "IN BOUNDS" if rw["in_bounds"] else ("no face contact" if not rw["face_hit"] else "OUT")
        cv2.putText(f, f"score {rw['total']:.2f}   {verdict}", (26, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2, cv2.LINE_AA)
        out.append(f)
    for _ in range(args.fps // 2):
        out.append(out[-1])

    imageio.mimsave(args.out, out, fps=args.fps, macro_block_size=1)
    print(f"\n{'stroke':24s} {'score':>7}  {'face':>5} {'in':>4}  landing")
    for name, rw, land in rows:
        lx = "none" if land is None else f"x={land[0]:.2f} y={land[1]:+.2f}"
        print(f"{name:24s} {rw['total']:7.2f}  {str(rw['face_hit']):>5} {str(rw['in_bounds']):>4}  {lx}")
    print(f"\nwrote {args.out} ({len(out)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
