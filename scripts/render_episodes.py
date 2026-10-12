"""Sample episodes end to end: feed in, oracle return, scored.

    .venv/bin/python scripts/render_episodes.py          # -> out/episodes.mp4

Each episode samples a strike point somewhere in the arm's underhand zone,
solves a feed that arrives there at rally speed from a randomised release,
asks the oracle for a return into the opponent's reachable zone, drives the
arm to that pose and swings. The score is computed from what actually happened
in contact, not from the plan - so a failed episode shows a real failure.
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
from sim.feed import _closest, _fly, aim_feed, sample_targets  # noqa: E402
from sim.oracle import plan_return  # noqa: E402
from sim.paddle import PADDLE_BODY, PaddleConfig  # noqa: E402
from sim.pose import solve_face_pose  # noqa: E402
from sim.reward import exchange_reward  # noqa: E402
from sim.zones import grid, sweep_face, zones  # noqa: E402

CFG = EnvConfig(paddle=PaddleConfig())
T = CFG.table
SWING = 0.34  # seconds from the wind-up pose to contact


def camera():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Framed on the striking half, not the whole table: the interesting part is
    # the paddle meeting the ball, and the far half is only context.
    cam.lookat[:] = (T.x_near + T.length * 0.30, 0.0, 0.10)
    cam.distance = 1.05
    cam.azimuth = 138.0
    cam.elevation = -14.0
    return cam


def paddle_geoms(model):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PADDLE_BODY)
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] == bid}


def run(world, feed, plan, hit_q, wind_q, fps, on_frame):
    """Fly the feed in and swing through it; return what the contact did."""
    model, data = world.model, world.data
    pads = paddle_geoms(model)
    ball_g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    ball_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BALL_BODY)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, BALL_BODY)
    qadr, vadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])

    mujoco.mj_resetData(model, data)
    data.qpos[world.qpos_adr] = wind_q
    world.command(wind_q)
    data.qpos[qadr:qadr + 3] = feed.pos
    data.qpos[qadr + 3:qadr + 7] = (1, 0, 0, 0)
    data.qvel[vadr:vadr + 6] = 0.0
    data.qvel[vadr:vadr + 3] = feed.vel
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    spf = max(1, round(1.0 / (fps * dt)))
    swing_start = max(0.0, feed.flight - SWING)

    contact_n = face_n = vel_out = landing = None
    prev_z = 1.0
    for k in range(int((feed.flight + 1.15) / dt)):
        t = k * dt
        a = 0.0 if t < swing_start else min(1.0, (t - swing_start) / SWING)
        world.command(wind_q + (hit_q - wind_q) * a)

        data.xfrc_applied[ball_b, :3] = drag_force(data.qvel[vadr:vadr + 3], CFG.ball.drag_k)
        mujoco.mj_step(model, data)

        if contact_n is None:
            for c in range(data.ncon):
                con = data.contact[c]
                pair = {con.geom1, con.geom2}
                if ball_g in pair and pair & pads:
                    contact_n = con.frame[:3].copy()
                    face_n = data.site_xmat[world.tip_site].reshape(3, 3)[:, 2].copy()
                    break
        elif vel_out is None and t > feed.flight + 0.03:
            vel_out = data.qvel[vadr:vadr + 3].copy()

        z = data.qpos[qadr + 2]
        if contact_n is not None and landing is None and prev_z > CFG.ball.radius >= z:
            landing = data.qpos[qadr:qadr + 2].copy()
        prev_z = z

        if k % spf == 0:
            on_frame()

    data.xfrc_applied[ball_b, :3] = 0.0
    return contact_n, face_n, vel_out, landing


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("out/episodes.mp4"))
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1100)
    p.add_argument("--height", type=int, default=660)
    p.add_argument("--seed", type=int, default=5)
    args = p.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    world = PingPongEnv(cfg=CFG, offscreen=(args.width, args.height))
    world.tip_site = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_SITE, "paddle_face")
    z = zones(world, T, swept=sweep_face(world))
    xe, ye = grid(T)

    strikes = sample_targets(z["underhand"], xe, ye, args.episodes * 3, seed=args.seed)
    lands = sample_targets(z["target"], xe, ye, args.episodes * 3, seed=args.seed + 1)

    cam = camera()
    images, rows = [], []
    made = 0

    with mujoco.Renderer(world.model, height=args.height, width=args.width) as r:
        for i, (s, l) in enumerate(zip(strikes, lands)):
            if made >= args.episodes:
                break
            feed = aim_feed(s, CFG.ball, spawn_x=(T.x_far - 0.25, T.x_far + 0.05),
                            table_e=T.restitution, seed=i)
            if feed.miss > 0.02:
                continue
            path, dt, bounce = _fly(feed.pos, feed.vel, CFG.ball.drag_k, CFG.ball.mass,
                                    CFG.ball.radius, T.restitution)
            d, j = _closest(path, s, after=bounce)
            j = max(1, min(j, len(path) - 1))
            v_in = (path[j] - path[j - 1]) / dt

            plan = plan_return(s, v_in, l, CFG.ball, T.restitution,
                               CFG.paddle.restitution, seed=i)
            hit = solve_face_pose(world, s, plan.normal)
            wind = solve_face_pose(world, s - np.array([0.0, 0.0, 0.045]), plan.normal)

            n_before = len(images)
            cn, fn, vo, land = run(
                world, feed, plan, hit.qpos, wind.qpos, args.fps,
                lambda: (r.update_scene(world.data, camera=cam), images.append(r.render())),
            )
            rw = exchange_reward(contact_normal=cn, face_normal=fn,
                                 ball_vel_out=vo if vo is not None else np.zeros(3),
                                 landing_xy=land, table=T, aim_xy=(l[0], l[1]))
            rows.append((n_before, len(images), made + 1, s, rw, plan))
            made += 1
            print(f"ep {made}: strike x={s[0]:+.2f} y={s[1]:+.2f} | "
                  f"plan {'OK ' if plan.feasible else 'HARD'} tilt {plan.tilt_deg:4.1f} deg | "
                  f"score {rw['total']:5.2f} {'IN' if rw['in_bounds'] else 'out'}")

    out = []
    for a, b, n, s, rw, plan in rows:
        verdict = "RETURNED IN BOUNDS" if rw["in_bounds"] else (
            "missed the table" if rw["face_hit"] else "no clean face contact")
        for f in images[a:b]:
            img = np.ascontiguousarray(f)
            cv2.putText(img, f"episode {n} of {len(rows)}", (26, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"strike x={s[0]:+.2f} y={s[1]:+.2f}   paddle tilt {plan.tilt_deg:.0f} deg",
                        (26, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (225, 225, 225), 1, cv2.LINE_AA)
            cv2.putText(img, f"score {rw['total']:.2f}   {verdict}", (26, 104),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 2, cv2.LINE_AA)
            out.append(img)
    for _ in range(args.fps // 2):
        out.append(out[-1])

    imageio.mimsave(args.out, out, fps=args.fps, macro_block_size=1)
    scored = [r[4] for r in rows]
    print(f"\n{sum(x['in_bounds'] for x in scored)}/{len(scored)} returned in bounds, "
          f"mean score {np.mean([x['total'] for x in scored]):.2f}")
    print(f"wrote {args.out} ({len(out)} frames, {len(out)/args.fps:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
