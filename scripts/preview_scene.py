"""Render the full two-arm scene to a video: table, both arms, launch zone.

This is a LOOK-AT-IT script, not part of the training stack. It exists so we
can see what the world actually contains before committing to a reward.

    .venv/bin/python scripts/preview_scene.py

Writes out/scene.mp4. No window needed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.arm import ARM_JOINTS, MODEL_DIR, SCENE  # noqa: E402
from sim.ball import add_ball, drag_force  # noqa: E402
from sim.config import EnvConfig  # noqa: E402
from sim.table import add_table  # noqa: E402

# Take physical numbers from the shared config so this preview tracks the real
# environment: measured table (31.5 x 49.5 in of waxed wood) and the 2 kHz step
# a stiff 2.7 g contact needs.
CFG = EnvConfig()
TABLE, BALL = CFG.table, CFG.ball


def sample_feed(seed: int):
    """Where a feed starts and how fast, sampled locally.

    Deliberately not sim.ball.sample_launch: this is a look-at-it script and
    should not break every time the training launcher's config is reshaped.
    """
    rng = np.random.default_rng(seed)
    pos = np.array([
        TABLE.length * rng.uniform(0.80, 0.93),
        rng.uniform(-0.18, 0.18),
        rng.uniform(0.10, 0.22),
    ])
    speed = rng.uniform(1.0, 1.5)
    elev = np.radians(rng.uniform(-2.0, 8.0))
    azim = np.pi + np.radians(rng.uniform(-6.0, 6.0))
    h = speed * np.cos(elev)
    vel = np.array([h * np.cos(azim), h * np.sin(azim), speed * np.sin(elev)])
    return pos, vel

READY_A = np.array([0.0, -0.75, 1.05, 0.55, 0.0])
READY_B = np.array([0.0, -0.75, 1.05, 0.55, 0.0])
N_MARKERS = 14


def build(offscreen):
    """Two SO-101s facing each other across the measured table, plus a ball."""
    spec = mujoco.MjSpec.from_file(str(SCENE))
    spec.option.timestep = CFG.timestep
    spec.visual.global_.offwidth, spec.visual.global_.offheight = offscreen

    # Arm B: same model, at the far end, turned to face back down -x.
    frame = spec.worldbody.add_frame()
    frame.pos = [TABLE.length, 0.0, 0.0]
    frame.quat = [0.0, 0.0, 0.0, 1.0]  # 180 deg about z
    spec.attach(mujoco.MjSpec.from_file(str(MODEL_DIR / "so101.xml")), prefix="b_", frame=frame)

    add_table(spec, TABLE)
    add_ball(spec, BALL, pos=(TABLE.length * 0.75, 0.0, 0.3))

    # Translucent markers showing where feeds get spawned from, and the target
    # zone on arm A's half that a return is supposed to land in.
    for i in range(N_MARKERS):
        p, _ = sample_feed(1000 + i)
        spec.worldbody.add_geom(
            name=f"launch_marker_{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=list(p), size=[0.012, 0, 0], contype=0, conaffinity=0,
            rgba=[1.0, 0.85, 0.1, 0.55],
        )
    spec.worldbody.add_geom(
        name="target_zone", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.32, 0.0, 0.002], size=[0.22, 0.26, 0.001],
        contype=0, conaffinity=0, rgba=[0.2, 0.7, 1.0, 0.30],
    )
    return spec.compile()


def camera(azimuth, distance=1.9, elevation=-22.0):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (TABLE.length / 2, 0.0, 0.05)
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("out/scene.mp4"))
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1100)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--launches", type=int, default=5)
    args = p.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model = build((args.width, args.height))
    data = mujoco.MjData(model)

    act = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)}
    ids_a = [act[n] for n in ARM_JOINTS]
    ids_b = [act["b_" + n] for n in ARM_JOINTS]
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball")
    qadr, vadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    ball_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

    def hold_ready():
        data.ctrl[ids_a] = READY_A
        data.ctrl[ids_b] = READY_B

    mujoco.mj_resetData(model, data)
    data.qpos[model.jnt_qposadr[[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]]] = READY_A
    data.qpos[model.jnt_qposadr[[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "b_" + n) for n in ARM_JOINTS]]] = READY_B
    hold_ready()
    mujoco.mj_forward(model, data)

    frames = []
    steps_per_frame = max(1, round(1.0 / (args.fps * model.opt.timestep)))

    with mujoco.Renderer(model, height=args.height, width=args.width) as r:
        # 1. Orbit the static scene so the objects are legible.
        for k in range(int(4.0 * args.fps)):
            hold_ready()
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
            r.update_scene(data, camera=camera(azimuth=110 + 200 * k / (4.0 * args.fps)))
            frames.append(r.render())
        print(f"orbit rendered ({len(frames)} frames)")

        # 2. Feed balls from sampled launch points, with drag.
        cam = camera(azimuth=150, distance=2.0, elevation=-16.0)
        for i in range(args.launches):
            lpos, lvel = sample_feed(2000 + i)
            data.qpos[qadr:qadr + 3] = lpos
            data.qpos[qadr + 3:qadr + 7] = (1, 0, 0, 0)
            data.qvel[vadr:vadr + 6] = 0.0
            data.qvel[vadr:vadr + 3] = lvel
            mujoco.mj_forward(model, data)

            bounce = None
            prev_vz = lvel[2]
            for _ in range(int(1.4 * args.fps)):
                hold_ready()
                for _ in range(steps_per_frame):
                    vel = data.qvel[vadr:vadr + 3]
                    data.xfrc_applied[ball_body, :3] = drag_force(vel, BALL.drag_k)
                    mujoco.mj_step(model, data)
                    vz = data.qvel[vadr + 2]
                    if bounce is None and prev_vz < 0 < vz:
                        bounce = data.qpos[qadr:qadr + 3].copy()
                    prev_vz = vz
                r.update_scene(data, camera=cam)
                frames.append(r.render())
            data.xfrc_applied[ball_body, :3] = 0.0
            where = "no bounce" if bounce is None else f"bounced at x={bounce[0]:.2f} y={bounce[1]:+.2f}"
            on_table = bounce is not None and 0 <= bounce[0] <= TABLE.length and abs(bounce[1]) <= TABLE.half_width
            print(f"feed {i}: from x={lpos[0]:.2f} y={lpos[1]:+.2f} z={lpos[2]:.2f} "
                  f"at {np.linalg.norm(lvel):.2f} m/s -> {where} "
                  f"{'(on table)' if on_table else '(OFF TABLE)'}")

    imageio.mimsave(args.out, frames, fps=args.fps, macro_block_size=1)
    print(f"\nwrote {args.out} ({len(frames)} frames, {len(frames)/args.fps:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
