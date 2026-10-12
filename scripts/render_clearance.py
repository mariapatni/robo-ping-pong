"""Sweep the legal pose set and show the blade never enters the table.

    .venv/bin/python scripts/render_clearance.py     # -> out/clearance.mp4

Every pose shown is drawn from the set the policy is allowed to choose: over
the table, inside the strike band, inside the underhand cone, and with the
blade rim clear of the surface. The bar on the right is the rim's height, so a
violation would be visible as the bar crossing zero. It never does, because the
set is filtered on exactly that quantity.
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
from sim.config import EnvConfig, TableConfig  # noqa: E402
from sim.env import PingPongEnv  # noqa: E402
from sim.paddle import PaddleConfig  # noqa: E402
from sim.pose import blade_lowest  # noqa: E402

W, H, PW = 660, 470, 340
T = TableConfig()
R = PaddleConfig().blade_diameter / 2
MARGIN = 0.005


def gauge(rim_mm, worst_mm, n, total):
    img = np.full((H, PW, 3), 250, np.uint8)
    cv2.putText(img, "blade rim height", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
    x0, y0, w, h = 110, 60, 86, H - 190
    top, bot = 140.0, -60.0

    def ypx(mm):
        return int(y0 + (top - mm) / (top - bot) * h)

    zero = ypx(0.0)
    cv2.rectangle(img, (x0, zero), (x0 + w, y0 + h), (205, 220, 235), -1)
    cv2.putText(img, "TABLE", (x0 + 14, zero + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    cv2.line(img, (x0 - 10, zero), (x0 + w + 10, zero), (40, 40, 40), 2)
    cv2.putText(img, "0", (x0 - 28, zero + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1)

    ym = ypx(MARGIN * 1000)
    cv2.line(img, (x0, ym), (x0 + w, ym), (70, 160, 70), 1, cv2.LINE_AA)
    cv2.putText(img, "5 mm margin", (x0 + w + 6, ym + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (70, 160, 70), 1)

    col = (60, 150, 60) if rim_mm >= 0 else (60, 60, 220)
    cv2.rectangle(img, (x0, ypx(rim_mm)), (x0 + w, zero), col, -1)
    cv2.putText(img, f"{rim_mm:+.0f} mm", (x0 + 6, ypx(rim_mm) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, col, 2, cv2.LINE_AA)

    y = H - 100
    for line in (f"pose {n} of {total}",
                 f"lowest seen so far: {worst_mm:+.1f} mm",
                 "violations: 0" if worst_mm >= 0 else f"VIOLATION {worst_mm:+.1f} mm"):
        cv2.putText(img, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (40, 40, 40), 1, cv2.LINE_AA)
        y += 26
    return img


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("out/clearance.mp4"))
    p.add_argument("--poses", type=int, default=120)
    p.add_argument("--fps", type=int, default=12)
    args = p.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    world = PingPongEnv(cfg=EnvConfig(paddle=PaddleConfig()), offscreen=(W, H))
    world.tip_site = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_SITE, "paddle_face")
    m = world.model
    probe = mujoco.MjData(m)

    N = (21, 13, 13, 11, 11)
    grids = [np.linspace(lo, hi, n) for lo, hi, n in zip(world.lower, world.upper, N)]
    mesh = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, 5)
    keep = []
    for q in mesh:
        probe.qpos[world.qpos_adr] = q
        mujoco.mj_kinematics(m, probe)
        pos = probe.site_xpos[world.tip_site]
        nrm = probe.site_xmat[world.tip_site].reshape(3, 3)[:, 2]
        if not (T.x_near <= pos[0] <= T.x_far and abs(pos[1]) <= T.half_width):
            continue
        if not (0.03 <= pos[2] <= 0.22):
            continue
        if nrm[2] < np.cos(np.radians(60)) or nrm[0] <= 0:
            continue
        if blade_lowest(pos, nrm, R) < MARGIN:
            continue
        keep.append(q)
    keep = np.array(keep)
    print(f"{len(keep):,} poses in the legal set; showing {args.poses}")

    rng = np.random.default_rng(1)
    show = keep[rng.choice(len(keep), min(args.poses, len(keep)), replace=False)]

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (T.x_near + 0.30, 0.0, 0.05)
    cam.distance, cam.elevation = 0.95, -6.0

    frames, worst = [], np.inf
    with mujoco.Renderer(m, height=H, width=W) as r:
        for i, q in enumerate(show, 1):
            world.reset(q)
            world.command(q)
            mujoco.mj_forward(m, world.data)
            pos = world.data.site_xpos[world.tip_site]
            nrm = world.data.site_xmat[world.tip_site].reshape(3, 3)[:, 2]
            rim = blade_lowest(pos, nrm, R) * 1000
            worst = min(worst, rim)
            cam.azimuth = 128.0 + 44.0 * np.sin(i / 17.0)
            r.update_scene(world.data, camera=cam)
            left = np.ascontiguousarray(r.render())
            cv2.rectangle(left, (0, 0), (W - 1, 30), (18, 18, 18), -1)
            cv2.putText(left, "every pose the policy is allowed to choose", (10, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, (245, 245, 245), 1, cv2.LINE_AA)
            panel = cv2.cvtColor(gauge(rim, worst, i, len(show)), cv2.COLOR_BGR2RGB)
            frames.append(np.hstack([left, panel]))

    for _ in range(args.fps):
        frames.append(frames[-1])
    imageio.mimsave(args.out, frames, fps=args.fps, macro_block_size=1)
    print(f"lowest rim over all {len(show)} poses: {worst:+.1f} mm")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
