"""The wrist camera's own view through a live episode, beside the scene view.

    .venv/bin/python scripts/render_wrist.py      # -> out/wrist_view.mp4

The wrist camera sits in the blade's plane, so the blade is a silhouette and a
ball shows up on the left of it or the right depending which face it is on.
This plays that against the third-person view so the two can be read together.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_episodes import CFG, T, camera, run  # noqa: E402

from sim.feed import _closest, _fly, aim_feed, sample_targets  # noqa: E402
from sim.env import PingPongEnv  # noqa: E402
from sim.oracle import plan_return  # noqa: E402
from sim.pose import solve_face_pose_ik, windup_pose  # noqa: E402
from sim.zones import grid, sweep_face, zones  # noqa: E402

W, H = 560, 440


def label(img, text, sub=None):
    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, 30), (18, 18, 18), -1)
    cv2.putText(img, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (230, 230, 230), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (90, 90, 90), 1)
    return img


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("out/wrist_view.mp4"))
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    world = PingPongEnv(cfg=CFG, offscreen=(W, H))
    world.tip_site = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_SITE, "paddle_face")
    z = zones(world, T, swept=sweep_face(world))
    xe, ye = grid(T)
    strikes = sample_targets(z["underhand"], xe, ye, args.episodes * 3, seed=5)
    lands = sample_targets(z["target"], xe, ye, args.episodes * 3, seed=6)

    cam = camera()
    cam.distance = 0.85
    frames, made = [], 0

    with mujoco.Renderer(world.model, height=H, width=W) as r:
        for i, (s, l) in enumerate(zip(strikes, lands)):
            if made >= args.episodes:
                break
            feed = aim_feed(s, CFG.ball, spawn_x=(T.x_far - 0.25, T.x_far + 0.05),
                            table_e=T.restitution, seed=i)
            if feed.miss > 0.02:
                continue
            path, dt, b = _fly(feed.pos, feed.vel, CFG.ball.drag_k, CFG.ball.mass,
                               CFG.ball.radius, T.restitution)
            d, j = _closest(path, s, after=b)
            j = max(1, min(j, len(path) - 1))
            plan = plan_return(s, (path[j] - path[j - 1]) / dt, l, CFG.ball,
                               T.restitution, CFG.paddle.restitution, seed=i)
            hit = solve_face_pose_ik(world, s, plan.normal, seed=i)
            wind = solve_face_pose_ik(world, windup_pose(s, plan.normal), plan.normal, seed=i)
            n = made + 1

            def grab():
                r.update_scene(world.data, camera=cam)
                scene = label(r.render(), f"scene view - episode {n}")
                r.update_scene(world.data, camera="wrist_cam")
                wrist = label(r.render(), "WRIST CAMERA - in the blade's plane",
                              "blade is the dark strip; ball left = front face, right = back")
                frames.append(np.hstack([scene, wrist]))

            run(world, feed, plan, hit.qpos, wind.qpos, args.fps, grab)
            made += 1
            print(f"episode {n}: strike x={s[0]:+.2f} y={s[1]:+.2f}, tilt {plan.tilt_deg:.0f} deg")

    for _ in range(args.fps // 2):
        frames.append(frames[-1])
    imageio.mimsave(args.out, frames, fps=args.fps, macro_block_size=1)
    print(f"\nwrote {args.out} ({len(frames)} frames, {len(frames)/args.fps:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
