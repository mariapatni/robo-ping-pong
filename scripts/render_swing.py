"""Show the swing against the ball: both paths, and the gap between them.

    .venv/bin/python scripts/render_swing.py      # -> out/swing.mp4

Left, the scene with both trails drawn. Right, the distance from paddle to
ball over time, so where the swing goes wrong is readable rather than inferred.
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
from render_episodes import CFG, SWING, T, paddle_geoms  # noqa: E402

from sim.ball import BALL_BODY, drag_force  # noqa: E402
from sim.env import PingPongEnv  # noqa: E402
from sim.feed import _closest, _fly, aim_feed, sample_targets  # noqa: E402
from sim.oracle import plan_return  # noqa: E402
from sim.pose import solve_face_pose_ik, windup_pose  # noqa: E402
from sim.zones import grid, sweep_face, zones  # noqa: E402

W, H = 620, 470
PW = 480
BLADE_R_MM = 55.0
USABLE_MM = 35.0


def plot(hist, k, title):
    """Paddle-to-ball distance over time, drawn up to sample k."""
    img = np.full((H, PW, 3), 250, np.uint8)
    cv2.putText(img, title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    x0, y0, w, h = 56, 54, PW - 80, H - 112
    top = 320.0

    def px(i, v):
        return (int(x0 + i / max(1, len(hist) - 1) * w), int(y0 + h - min(v, top) / top * h))

    for mm, col, lab in ((USABLE_MM, (70, 150, 70), "35 mm: clean face hit"),
                         (BLADE_R_MM, (90, 90, 210), "55 mm: blade edge")):
        y = int(y0 + h - mm / top * h)
        cv2.line(img, (x0, y), (x0 + w, y), col, 1, cv2.LINE_AA)
        cv2.putText(img, lab, (x0 + 6, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)

    cv2.line(img, (x0, y0), (x0, y0 + h), (120, 120, 120), 1)
    cv2.line(img, (x0, y0 + h), (x0 + w, y0 + h), (120, 120, 120), 1)
    cv2.putText(img, "0", (x0 - 20, y0 + h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)
    cv2.putText(img, "320mm", (x0 - 50, y0 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

    pts = [px(i, v) for i, v in enumerate(hist[: k + 1])]
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, (200, 60, 40), 2, cv2.LINE_AA)
    if pts:
        cv2.circle(img, pts[-1], 5, (200, 60, 40), -1)
        cv2.putText(img, f"{hist[k]:.0f} mm", (pts[-1][0] + 8, pts[-1][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 60, 40), 1, cv2.LINE_AA)
    cv2.putText(img, "paddle-to-ball distance", (x0, H - 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(img, "the dip is the swing passing the ball", (x0, H - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
    return img


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("out/swing.mp4"))
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    world = PingPongEnv(cfg=CFG, offscreen=(W, H))
    world.tip_site = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_SITE, "paddle_face")
    z = zones(world, T, swept=sweep_face(world))
    xe, ye = grid(T)
    S = sample_targets(z["underhand"], xe, ye, 20, seed=5)
    L = sample_targets(z["target"], xe, ye, 20, seed=6)

    m, d = world.model, world.data
    pads = paddle_geoms(m)
    bg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    bb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, BALL_BODY)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, BALL_BODY)
    qa, va = int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 0.78, 142.0, -12.0

    frames, made = [], 0
    with mujoco.Renderer(m, height=H, width=W) as r:
        for i, (s, l) in enumerate(zip(S, L)):
            if made >= args.episodes:
                break
            f = aim_feed(s, CFG.ball, spawn_x=(T.x_far - 0.25, T.x_far + 0.05),
                         table_e=T.restitution, seed=i)
            if f.miss > 0.02:
                continue
            path, dt0, b = _fly(f.pos, f.vel, CFG.ball.drag_k, CFG.ball.mass,
                                CFG.ball.radius, T.restitution)
            _, j = _closest(path, s, after=b)
            j = max(1, min(j, len(path) - 1))
            pl = plan_return(s, (path[j] - path[j - 1]) / dt0, l, CFG.ball,
                             T.restitution, CFG.paddle.restitution, seed=i)
            hit = solve_face_pose_ik(world, s, pl.normal, seed=i)
            wind = solve_face_pose_ik(world, windup_pose(s, pl.normal), pl.normal, seed=i)

            mujoco.mj_resetData(m, d)
            d.qpos[world.qpos_adr] = wind.qpos
            world.command(wind.qpos)
            d.qpos[qa:qa + 3] = f.pos
            d.qpos[qa + 3:qa + 7] = (1, 0, 0, 0)
            d.qvel[va:va + 6] = 0.0
            d.qvel[va:va + 3] = f.vel
            mujoco.mj_forward(m, d)

            dt = m.opt.timestep
            spf = max(1, round(1.0 / (args.fps * dt)))
            ss = max(0.0, f.flight - SWING)
            shots, gaps, touched = [], [], False
            for k in range(int((f.flight + 0.3) / dt)):
                t = k * dt
                a = 0.0 if t < ss else min(1.0, (t - ss) / SWING)
                world.command(wind.qpos + (hit.qpos - wind.qpos) * a)
                d.xfrc_applied[bb, :3] = drag_force(d.qvel[va:va + 3], CFG.ball.drag_k)
                mujoco.mj_step(m, d)
                if not touched:
                    for c in range(d.ncon):
                        pr = {d.contact[c].geom1, d.contact[c].geom2}
                        if bg in pr and pr & pads:
                            touched = True
                            break
                if k % spf == 0:
                    r.update_scene(d, camera=cam)
                    shots.append(r.render())
                    gaps.append(float(np.linalg.norm(d.qpos[qa:qa + 3] - d.site_xpos[world.tip_site])) * 1000)
            d.xfrc_applied[bb, :3] = 0.0
            made += 1
            verdict = "CONTACT" if touched else "MISSED"
            low = min(gaps)
            print(f"episode {made}: closest {low:5.1f} mm   {verdict}")
            for k, shot in enumerate(shots):
                left = np.ascontiguousarray(shot)
                cv2.rectangle(left, (0, 0), (W - 1, 30), (18, 18, 18), -1)
                cv2.putText(left, f"episode {made}   {verdict} (closest {low:.0f} mm)", (10, 21),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.47, (245, 245, 245), 1, cv2.LINE_AA)
                if not touched:
                    cv2.rectangle(left, (0, 0), (W - 1, H - 1), (255, 90, 70), 3)
                # MuJoCo renders RGB; the plot is drawn with OpenCV in BGR.
                # Convert only the plot, rather than the whole frame.
                panel = cv2.cvtColor(plot(gaps, k, "does the paddle reach the ball?"),
                                     cv2.COLOR_BGR2RGB)
                frames.append(np.hstack([left, panel]))

    for _ in range(args.fps // 2):
        frames.append(frames[-1])
    imageio.mimsave(args.out, frames, fps=args.fps, macro_block_size=1)
    print(f"\nwrote {args.out} ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
