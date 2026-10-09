"""What each camera actually sees: both arms, both viewpoints, one rally feed.

    .venv/bin/python scripts/camera_views.py      # -> out/camera_views_quad.mp4

A 2x2 grid - each arm's fixed table-side camera and its wrist camera - so the
inputs the perception stack will really get are visible side by side. The fixed
cameras carry the real OV9281 intrinsics (1.8 mm lens, 3.84 x 2.40 mm sensor,
640x400), so their framing is what the bought hardware will frame.
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
from sim.arm import ARM_JOINTS, Arm, MODEL_DIR, SCENE  # noqa: E402
from sim.ball import BALL_BODY, add_ball, drag_force  # noqa: E402
from sim.cameras import add_cameras  # noqa: E402
from sim.config import EnvConfig, TableConfig  # noqa: E402
from sim.feed import aim_feed, sample_targets  # noqa: E402
from sim.paddle import PaddleConfig, add_paddle  # noqa: E402
from sim.table import add_corner_markers, add_net, add_table  # noqa: E402
from sim.zones import grid, sweep_face, zones  # noqa: E402

CFG = EnvConfig(paddle=PaddleConfig())
T = CFG.table
READY = np.array([0.0, -0.75, 1.05, 0.55, 0.0])
VIEW = (512, 320)


def build(offscreen):
    spec = mujoco.MjSpec.from_file(str(SCENE))
    spec.option.timestep = CFG.timestep
    spec.visual.global_.offwidth, spec.visual.global_.offheight = offscreen

    frame = spec.worldbody.add_frame()
    frame.pos = [T.x_far - T.arm_inset, 0.0, 0.0]
    frame.quat = [0.0, 0.0, 0.0, 1.0]  # 180 deg about z: a rotation, not a mirror
    spec.attach(mujoco.MjSpec.from_file(str(MODEL_DIR / "so101.xml")), prefix="b_", frame=frame)

    add_table(spec, T, CFG.surface)
    add_corner_markers(spec, T, CFG.markers)
    add_net(spec, T, CFG.net)
    add_cameras(spec, T, CFG.cameras)
    add_paddle(spec, CFG.paddle)
    add_paddle(spec, CFG.paddle, prefix="b_")
    add_ball(spec, CFG.ball, pos=(T.x_far * 0.6, 0.0, 0.25))
    return spec.compile()


def tile(frames, labels, size):
    cells = []
    for img, lab in zip(frames, labels):
        f = np.ascontiguousarray(img)
        cv2.rectangle(f, (0, 0), (f.shape[1] - 1, 26), (18, 18, 18), -1)
        cv2.putText(f, lab, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.rectangle(f, (0, 0), (f.shape[1] - 1, f.shape[0] - 1), (90, 90, 90), 1)
        cells.append(f)
    return np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:])])


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("out/camera_views_quad.mp4"))
    p.add_argument("--feeds", type=int, default=4)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model = build((VIEW[0], VIEW[1]))
    data = mujoco.MjData(model)

    act = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)}
    ids_a = [act[n] for n in ARM_JOINTS]
    ids_b = [act["b_" + n] for n in ARM_JOINTS]
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, BALL_BODY)
    qadr, vadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    bb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BALL_BODY)

    cams = ["cam_a", "wrist_cam", "cam_b", "b_wrist_cam"]  # unprefixed arm is A
    labels = ["ARM A - fixed table camera", "ARM A - wrist camera",
              "ARM B - fixed table camera", "ARM B - wrist camera"]
    have = []
    for c in cams:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c)
        have.append(cid)
    missing = [c for c, i in zip(cams, have) if i < 0]
    if missing:
        raise SystemExit(f"cameras not in model: {missing}")

    # Feeds aimed across arm A's strike zone, so the views show real variety.
    probe = Arm(modify=lambda s: add_paddle(s, CFG.paddle))
    probe.tip_site = mujoco.mj_name2id(probe.model, mujoco.mjtObj.mjOBJ_SITE, "paddle_face")
    zz = zones(probe, T, swept=sweep_face(probe, samples=(21, 11, 11, 9, 9)))
    xe, ye = grid(T)
    targets = sample_targets(zz["underhand"], xe, ye, args.feeds, seed=31)

    mujoco.mj_resetData(model, data)
    for ids in (ids_a, ids_b):
        data.ctrl[ids] = READY
    frames = []
    dt = model.opt.timestep
    spf = max(1, round(1.0 / (args.fps * dt)))

    with mujoco.Renderer(model, height=VIEW[1], width=VIEW[0]) as r:
        for i, tgt in enumerate(targets):
            f = aim_feed(tgt, CFG.ball, spawn_x=(T.x_far - 0.25, T.x_far + 0.05),
                         table_e=T.restitution, seed=100 + i)
            data.qpos[qadr:qadr + 3] = f.pos
            data.qpos[qadr + 3:qadr + 7] = (1, 0, 0, 0)
            data.qvel[vadr:vadr + 6] = 0.0
            data.qvel[vadr:vadr + 3] = f.vel
            mujoco.mj_forward(model, data)
            for _ in range(int(1.3 / dt / spf)):
                for ids in (ids_a, ids_b):
                    data.ctrl[ids] = READY
                for _ in range(spf):
                    data.xfrc_applied[bb, :3] = drag_force(data.qvel[vadr:vadr + 3], CFG.ball.drag_k)
                    mujoco.mj_step(model, data)
                views = []
                for c in cams:
                    r.update_scene(data, camera=c)
                    views.append(r.render())
                frames.append(tile(views, labels, VIEW))
            print(f"feed {i + 1}: target x={tgt[0]:+.2f} y={tgt[1]:+.2f}, aim error {f.miss * 1000:.1f} mm")

    imageio.mimsave(args.out, frames, fps=args.fps, macro_block_size=1)
    print(f"\nwrote {args.out} ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
