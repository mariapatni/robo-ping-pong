"""Run many episodes and attribute every failure to a cause.

    .venv/bin/python scripts/evaluate.py --episodes 200

Several fixes so far looked convincing on eight episodes and vanished at
thirty, so this exists to stop that: enough episodes for the error bars to be
smaller than the effects, and a labelled cause for each failure rather than a
single pass rate. A rate tells you that something is wrong; a breakdown tells
you what to work on next.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.ball import BALL_BODY, drag_force  # noqa: E402
from sim.config import EnvConfig  # noqa: E402
from sim.env import PingPongEnv  # noqa: E402
from sim.feed import _closest, _fly, aim_feed, sample_targets  # noqa: E402
from sim.oracle import plan_return, plan_within_reach  # noqa: E402
from sim.paddle import PADDLE_BODY, PaddleConfig  # noqa: E402
from sim.pose import solve_face_pose_ik, windup_pose  # noqa: E402
from sim.reward import exchange_reward  # noqa: E402
from sim.safety import TableGuard  # noqa: E402
from sim.zones import grid, sweep_face, zones  # noqa: E402

CFG = EnvConfig(paddle=PaddleConfig())
T = CFG.table
R = CFG.paddle.blade_diameter / 2
SWING = 0.34


def classify(rec):
    """Why this episode did not score, in one word."""
    if rec["declined"]:
        return "declined: no legal pose"
    if not rec["contacted"]:
        return "miss: paddle never reached the ball"
    if not rec["face_hit"]:
        return "miss: caught the rim"
    if rec["in_bounds"]:
        return "returned"
    land = rec["landing"]
    if land is None:
        return "out: ball never came down"
    if land[0] < T.x_net:
        return "out: into its own half"
    if land[0] > T.x_far:
        return "out: past the far edge"
    return "out: wide"


def run_episode(world, guard, s, l, i, planner):
    feed = aim_feed(s, CFG.ball, spawn_x=(T.x_far - 0.25, T.x_far + 0.05),
                    table_e=T.restitution, seed=i)
    if feed.miss > 0.02:
        return None

    path, dt0, b = _fly(feed.pos, feed.vel, CFG.ball.drag_k, CFG.ball.mass,
                        CFG.ball.radius, T.restitution)
    _, j = _closest(path, s, after=b)
    j = max(1, min(j, len(path) - 1))
    v_in = (path[j] - path[j - 1]) / dt0

    if planner == "reach":
        plan, hit = plan_within_reach(world, guard, s, v_in, l, CFG.ball,
                                      T.restitution, CFG.paddle.restitution, seed=i)
        if plan is None:
            return {"declined": True, "contacted": False, "face_hit": False,
                    "in_bounds": False, "landing": None, "total": 0.0,
                    "tilt": float("nan"), "strike": s, "penetration": 0.0}
    else:
        plan = plan_return(s, v_in, l, CFG.ball, T.restitution,
                           CFG.paddle.restitution, seed=i)
        hit = solve_face_pose_ik(world, s, plan.normal, seed=i, guard=guard)

    wind = solve_face_pose_ik(world, windup_pose(s, plan.normal, blade_radius=R),
                              plan.normal, seed=i, guard=guard)

    m, d = world.model, world.data
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PADDLE_BODY)
    pads = {g for g in range(m.ngeom) if m.geom_bodyid[g] == bid}
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
    bg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    bb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, BALL_BODY)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, BALL_BODY)
    qa, va = int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])

    mujoco.mj_resetData(m, d)
    d.qpos[world.qpos_adr] = wind.qpos
    guard.command(wind.qpos)
    d.qpos[qa:qa + 3] = feed.pos
    d.qpos[qa + 3:qa + 7] = (1, 0, 0, 0)
    d.qvel[va:va + 6] = 0.0
    d.qvel[va:va + 3] = feed.vel
    mujoco.mj_forward(m, d)

    dt = m.opt.timestep
    start = max(0.0, feed.flight - SWING)
    cn = fn = vo = land = None
    prev_z, pen = 1.0, 0.0
    for k in range(int((feed.flight + 1.0) / dt)):
        t = k * dt
        a = 0.0 if t < start else min(1.0, (t - start) / SWING)
        guard.command(wind.qpos + (hit.qpos - wind.qpos) * a)
        d.xfrc_applied[bb, :3] = drag_force(d.qvel[va:va + 3], CFG.ball.drag_k)
        mujoco.mj_step(m, d)
        for c in range(d.ncon):
            con = d.contact[c]
            pair = {con.geom1, con.geom2}
            if tid in pair and (pair & guard.arm_geoms):
                pen = min(pen, float(con.dist))
        if cn is None:
            for c in range(d.ncon):
                pair = {d.contact[c].geom1, d.contact[c].geom2}
                if bg in pair and pair & pads:
                    cn = d.contact[c].frame[:3].copy()
                    fn = d.site_xmat[world.tip_site].reshape(3, 3)[:, 2].copy()
                    break
        elif vo is None and t > feed.flight + 0.03:
            vo = d.qvel[va:va + 3].copy()
        z = d.qpos[qa + 2]
        if cn is not None and land is None and prev_z > CFG.ball.radius >= z:
            land = d.qpos[qa:qa + 2].copy()
        prev_z = z
    d.xfrc_applied[bb, :3] = 0.0

    r = exchange_reward(contact_normal=cn, face_normal=fn,
                        ball_vel_out=vo if vo is not None else np.zeros(3),
                        landing_xy=land, table=T, aim_xy=(l[0], l[1]),
                        blade_lowest=pen)
    return {"declined": False, "contacted": cn is not None, "face_hit": r["face_hit"],
            "in_bounds": r["in_bounds"], "landing": None if land is None else land.tolist(),
            "total": r["total"], "tilt": plan.tilt_deg, "strike": s.tolist(),
            "penetration": pen}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--planner", choices=["reach", "demand"], default="reach")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--out", type=Path, default=Path("out/evaluation.json"))
    args = p.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    world = PingPongEnv(cfg=CFG)
    world.tip_site = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_SITE, "paddle_face")
    guard = TableGuard(world)
    z = zones(world, T, swept=sweep_face(world))
    xe, ye = grid(T)

    n = args.episodes
    strikes = sample_targets(z["underhand"], xe, ye, n * 2, seed=args.seed)
    lands = sample_targets(z["target"], xe, ye, n * 2, seed=args.seed + 1)

    records, causes = [], Counter()
    for i, (s, l) in enumerate(zip(strikes, lands)):
        if len(records) >= n:
            break
        rec = run_episode(world, guard, s, l, i, args.planner)
        if rec is None:
            continue
        rec["cause"] = classify(rec)
        causes[rec["cause"]] += 1
        records.append(rec)
        if len(records) % 25 == 0:
            done = sum(1 for r in records if r["in_bounds"])
            print(f"  {len(records):4d} episodes, {done / len(records) * 100:4.1f}% returned", flush=True)

    attempted = [r for r in records if not r["declined"]]
    returned = [r for r in records if r["in_bounds"]]
    pen = np.array([r["penetration"] for r in records]) * 1000

    print(f"\n{len(records)} episodes, planner = {args.planner}\n")
    print(f"  returned in bounds      {len(returned) / len(records) * 100:5.1f}%  of all episodes")
    if attempted:
        print(f"                          {len(returned) / len(attempted) * 100:5.1f}%  of those attempted")
    print(f"  declined as unreachable {sum(r['declined'] for r in records) / len(records) * 100:5.1f}%")
    print(f"  touched the table       {(pen < -0.5).mean() * 100:5.1f}%   deepest {pen.min():+.2f} mm")

    print(f"\n  {'cause':42} {'count':>6} {'share':>7}")
    for cause, c in causes.most_common():
        print(f"  {cause:42} {c:6d} {c / len(records) * 100:6.1f}%")

    # 95% interval on the headline rate, so a future comparison knows what is noise.
    pr = len(returned) / len(records)
    se = np.sqrt(pr * (1 - pr) / len(records))
    print(f"\n  return rate {pr * 100:.1f}% +/- {1.96 * se * 100:.1f} (95%), n={len(records)}")

    args.out.write_text(json.dumps({"planner": args.planner, "records": records,
                                    "causes": dict(causes)}, indent=1))
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
