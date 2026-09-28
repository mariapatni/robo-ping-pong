"""Where should the two fixed cameras go? Measure it, then look at it.

The plan under test: no wrist cameras; one fixed USB camera per side, mounted
BEHIND and ABOVE its own arm and angled down, the two fused by a single Kalman
filter into one ball estimate in the table frame. The obvious risk with mounting
behind the arm is that the arm is then between the camera and the ball exactly
when the ball matters most, so that is what this script measures.

    .venv/bin/python scripts/camera_study.py              # sweep + montage
    .venv/bin/python scripts/camera_study.py --video      # + a rally, both feeds

For every candidate pose it reports:

  coverage     fraction of the table surface inside the frame, per side
  visible      fraction of the ball that is in frame and unobstructed, averaged
               over real launcher rollouts with both arms tracking the ball
  both         fraction of ball samples BOTH cameras can see - the only samples
               the Kalman filter gets a 3D fix from
  px           pixels across the 40 mm ball
  parallax     angle between the two view rays; 90 deg is ideal, 0 or 180 is the
               degenerate two-cameras-staring-at-each-other case
  sigma        one-sigma position error on the worst axis, from 0.5 px bearing
               noise on both cameras

Writes out/camera_study.png (the montage), out/camera_study.txt (the table), and
with --video out/camera_views.mp4.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.arm import ARM_JOINTS, MODEL_DIR, SCENE  # noqa: E402
from sim.ball import BALL_BODY, add_ball, drag_force, sample_launch  # noqa: E402
from sim.cameras import (  # noqa: E402
    Pinhole,
    add_camera,
    add_cameras,
    camera_pose,
    coverage,
    parallax_deg,
    triangulation_sigma,
    visible_fraction,
)
from sim.config import CameraConfig, CameraRig, EnvConfig  # noqa: E402
from sim.table import add_corner_markers, add_net, add_table  # noqa: E402

CFG = EnvConfig()
TABLE, BALL, NET = CFG.table, CFG.ball, CFG.net

# Poses the arms hold and reach through while the ball comes at them. Not a
# policy - just enough motion that the arm is in the way when a real one would
# be, which is the whole point of the occlusion measurement.
READY = np.array([0.0, -0.75, 1.05, 0.55, 0.0])
STRIKE = np.array([0.0, -0.30, 0.70, 0.30, 0.0])

# The sweep. Deliberately coarse and wide: the question is which region of the
# space works, not the third decimal place of one pose.
HEIGHTS = (0.15, 0.25, 0.35, 0.45, 0.60)
BACKS = (0.0, 0.15, 0.30, 0.50)
PITCHES = (20.0, 30.0, 40.0, 50.0, 60.0)
LATERALS = (0.0, 0.25, 0.40)
# Yaw is toward the camera's own left, so a NEGATIVE yaw swings a laterally
# offset camera back across the table - which is the only way an offset mount
# keeps the near corner on its own side in frame.
YAWS = (0.0, -10.0, -20.0)
FOCALS = (1.4, 1.8, 2.1, 2.8, 3.6)  # M12 lenses you can actually buy


# -- the world ----------------------------------------------------------------


def build(extra: list[Pinhole] | None = None, offscreen=(1280, 800)):
    """Two SO-101s facing each other across the dressed table, plus a ball.

    Both arms are present because the thing being measured is self-occlusion:
    a single-arm world would flatter every candidate on the far side.
    """
    spec = mujoco.MjSpec.from_file(str(SCENE))
    spec.option.timestep = CFG.timestep
    spec.visual.global_.offwidth, spec.visual.global_.offheight = offscreen

    frame = spec.worldbody.add_frame()
    frame.pos = [TABLE.length, 0.0, 0.0]
    frame.quat = [0.0, 0.0, 0.0, 1.0]  # 180 deg about z
    spec.attach(mujoco.MjSpec.from_file(str(MODEL_DIR / "so101.xml")), prefix="b_", frame=frame)

    add_table(spec, TABLE, CFG.surface)
    add_corner_markers(spec, TABLE, CFG.markers)
    add_net(spec, TABLE, NET)
    add_cameras(spec, TABLE, CFG.cameras)
    for i, cam in enumerate(extra or []):
        add_camera(spec, cam, f"study_{i:02d}")
    add_ball(spec, BALL, pos=(0.75 * TABLE.length, 0.0, 0.3))
    return spec.compile()


class World:
    """The compiled two-arm scene plus the handles to drive it."""

    def __init__(self, extra=None):
        self.model = build(extra)
        self.data = mujoco.MjData(self.model)
        name2id = mujoco.mj_name2id
        act = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
               for i in range(self.model.nu)}
        self.ids_a = [act[n] for n in ARM_JOINTS]
        self.ids_b = [act["b_" + n] for n in ARM_JOINTS]
        jid = name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, BALL_BODY)
        self.qadr = int(self.model.jnt_qposadr[jid])
        self.vadr = int(self.model.jnt_dofadr[jid])
        self.ball_body = name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, BALL_BODY)
        self.qpos_a = self.model.jnt_qposadr[
            [name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]]
        self.qpos_b = self.model.jnt_qposadr[
            [name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "b_" + n) for n in ARM_JOINTS]]

    def ball(self):
        return self.data.qpos[self.qadr:self.qadr + 3].copy()

    def track(self):
        """Point each arm at the ball and extend as it arrives.

        Crude on purpose: pan at the ball, and blend from the ready pose toward a
        reaching pose as the ball closes on the arm's own baseline. It puts the
        arm where a real one would be - upright and central at rest, stretched
        out across its own half at contact - which is the geometry occlusion
        depends on.
        """
        b = self.ball()
        for ids, (bx, by) in ((self.ids_a, (b[0], b[1])),
                              (self.ids_b, (TABLE.length - b[0], -b[1]))):
            alpha = float(np.clip((0.60 - bx) / 0.40, 0.0, 1.0))
            q = READY + alpha * (STRIKE - READY)
            q[0] = np.clip(np.arctan2(by, max(bx, 0.05)), -1.4, 1.4)
            self.data.ctrl[ids] = q


def rollouts(world: World, seeds, dt: float = 0.02, seconds: float = 2.2):
    """Run the scripted feeder and snapshot (ball, whole-model pose) as it goes.

    Snapshots are whole qpos vectors so a candidate camera can be scored later
    against the exact configuration the arms were in, without re-simulating once
    per candidate.
    """
    m, d = world.model, world.data
    per = max(1, round(dt / m.opt.timestep))
    out = []
    for seed in seeds:
        feed = sample_launch(CFG.launch, seed, TABLE.length)
        mujoco.mj_resetData(m, d)
        d.qpos[world.qpos_a] = READY
        d.qpos[world.qpos_b] = READY
        d.qpos[world.qadr:world.qadr + 3] = feed.pos
        d.qpos[world.qadr + 3:world.qadr + 7] = (1, 0, 0, 0)
        d.qvel[world.vadr:world.vadr + 3] = feed.vel
        mujoco.mj_forward(m, d)
        for _ in range(int(seconds / dt)):
            world.track()
            for _ in range(per):
                d.xfrc_applied[world.ball_body, :3] = drag_force(
                    d.qvel[world.vadr:world.vadr + 3], BALL.drag_k)
                mujoco.mj_step(m, d)
            b = world.ball()
            if b[2] < -0.05 or b[0] < 0.05:
                # Off the table, or past the arm's baseline: the point is dead
                # and nobody needs to be tracking it any more.
                break
            # Only count the ball while it is over the table and in play: a ball
            # rolling on the floor is not something the tracker has to see.
            if (-0.05 <= b[0] <= TABLE.length + 0.05
                    and abs(b[1]) <= TABLE.half_width + 0.05 and b[2] > 0.005):
                out.append((seed, b, d.qpos.copy()))
        d.xfrc_applied[:] = 0.0
    return out


# -- scoring ------------------------------------------------------------------


# The window a return actually has to be tracked through: over the arm's own
# half, out to the depth it can reach, in the band a paddle can meet the ball.
STRIKE_X = 0.48
STRIKE_BAND = (0.03, 0.22)


def in_strike_zone(ball) -> bool:
    return bool(ball[0] <= STRIKE_X and STRIKE_BAND[0] <= ball[2] <= STRIKE_BAND[1])


def score(world: World, samples, cfg: CameraConfig, noise_px: float = 0.5,
          dt: float = 0.04) -> dict:
    """Every number for one candidate, as a symmetric pair of cameras.

    Two of these matter more than the averages. `strike` is visibility restricted
    to the strike zone, where a lost ball actually costs a return. `blackout` is
    the longest unbroken run of frames in which camera A cannot see the ball:
    a Kalman filter coasts happily through one dropped frame and diverges
    through fifteen, so the length of the gap matters more than its rate.
    """
    a = camera_pose(cfg, TABLE, "a")
    b = camera_pose(cfg, TABLE, "b")
    m, d = world.model, world.data

    vis = {"a": [], "b": []}
    px, par, sig, both = [], [], [], []
    near = {"a": [], "b": []}
    strike, blackout, run, prev_seed = [], 0, 0, None
    fused_gap, fused_run = 0, 0
    for _seed, ball, qpos in samples:
        if _seed != prev_seed:
            run, fused_run, prev_seed = 0, 0, _seed
        d.qpos[:] = qpos
        mujoco.mj_kinematics(m, d)
        va = visible_fraction(m, d, a, ball, BALL.radius, world.ball_body)
        vb = visible_fraction(m, d, b, ball, BALL.radius, world.ball_body)
        vis["a"].append(va)
        vis["b"].append(vb)
        # "Its own half" is where each arm is actually working, and where its own
        # camera is most likely to be looking straight through it.
        (near["a"] if ball[0] < TABLE.half_length else near["b"]).append(
            va if ball[0] < TABLE.half_length else vb)
        if in_strike_zone(ball):
            strike.append(va)
        run = run + 1 if va < 0.5 else 0
        blackout = max(blackout, run)
        # The gap that actually hurts is the one where NEITHER camera has the
        # ball, because that is the only time the filter has nothing to eat.
        fused_run = fused_run + 1 if max(va, vb) < 0.5 else 0
        fused_gap = max(fused_gap, fused_run)
        seen = va >= 0.5 and vb >= 0.5
        both.append(seen)
        if seen:
            px.append(min(a.pixels_across(ball, 2 * BALL.radius),
                          b.pixels_across(ball, 2 * BALL.radius)))
            par.append(parallax_deg(a, b, ball))
            sig.append(triangulation_sigma([a, b], ball, noise_px)[0])

    def med(v, default=float("nan")):
        return float(np.median(v)) if len(v) else default

    # "The whole table is in frame" means every corner, not 99% of the area:
    # the corners carry the calibration markers, and a corner that drops out is
    # a corner the extrinsics cannot be recovered from.
    corners = all(cam.in_frame(np.array([x, y, 0.0]))
                  for cam in (a, b)
                  for x in (0.0, TABLE.length)
                  for y in (-TABLE.half_width, TABLE.half_width))
    return {
        "cfg": cfg,
        "corners": corners,
        "cov_a": coverage(a, TABLE),
        "cov_b": coverage(b, TABLE),
        "vis_a": float(np.mean(vis["a"])),
        "vis_b": float(np.mean(vis["b"])),
        "own_a": float(np.mean(near["a"])) if near["a"] else float("nan"),
        "own_b": float(np.mean(near["b"])) if near["b"] else float("nan"),
        "both": float(np.mean(both)),
        "strike": float(np.mean(strike)) if strike else float("nan"),
        "blackout_ms": 1000.0 * dt * blackout,
        "fused_gap_ms": 1000.0 * dt * fused_gap,
        "par_bad": float(np.mean(np.array(par) < 20.0)) if par else float("nan"),
        "px": med(px, 0.0),
        "par": med(par),
        "par_max": float(np.max(par)) if par else float("nan"),
        "sigma_mm": 1000.0 * med(sig, float("inf")),
        "sigma_p95_mm": 1000.0 * float(np.percentile(sig, 95)) if sig else float("inf"),
        "n": len(samples),
    }


def RANK(r: dict):
    """Rank order, worst-case first.

    Longest fused blackout is the thing that breaks a tracker, so it leads - but
    binned to 100 ms, because one sample either way at 40 ms per frame is noise,
    not a preference. Then the fraction of the flight with a two-camera fix, then
    the triangulated position error.
    """
    return (int(r["fused_gap_ms"] // 100), -r["both"], r["sigma_mm"])


def label(cfg: CameraConfig) -> str:
    return (f"h{cfg.height:.2f} back{cfg.back:.2f} "
            f"pitch{cfg.pitch_deg:.0f} yaw{cfg.yaw_deg:+.0f} lat{cfg.lateral:.2f} "
            f"f{cfg.focal_mm:.1f}")


def row(r: dict) -> str:
    return (f"{label(r['cfg']):<52} cov {r['cov_a']:.2f}/{r['cov_b']:.2f}"
            f"{'+' if r['corners'] else '-'} "
            f"vis {r['vis_a']:.2f}/{r['vis_b']:.2f}  own {r['own_a']:.2f}/{r['own_b']:.2f}  "
            f"strike {r['strike']:.2f}  gapA {r['blackout_ms']:4.0f} fused {r['fused_gap_ms']:4.0f}ms  "
            f"both {r['both']:.2f}  px {r['px']:4.1f}  "
            f"par {r['par']:5.1f} (max {r['par_max']:5.1f}, <20deg {r['par_bad']:.2f})  "
            f"sig {r['sigma_mm']:5.1f}/{r['sigma_p95_mm']:6.1f} mm")


# -- the montage --------------------------------------------------------------


def montage(cands: list[CameraConfig], samples, path: Path, cols: int = 3, tag=None):
    """Render each candidate's own view of the same moment and tile them.

    The moment chosen is the one where the ball is deepest into arm A's half and
    still above the table: the instant the arm is most likely to be standing in
    front of its own camera.
    """
    pick = min((s for s in samples if s[1][2] > 0.02), key=lambda s: s[1][0])
    cams = [camera_pose(c, TABLE, "a") for c in cands]
    world = World(extra=cams)
    world.data.qpos[:] = pick[2]
    mujoco.mj_forward(world.model, world.data)

    w, h = cands[0].resolution
    tiles = []
    with mujoco.Renderer(world.model, h, w) as r:
        for i, (cfg, cam) in enumerate(zip(cands, cams)):
            r.update_scene(world.data, camera=f"study_{i:02d}")
            img = Image.fromarray(r.render())
            draw = ImageDraw.Draw(img)
            # Ring the ball where the maths says it is, so a reader can tell
            # "not visible" from "I cannot find it".
            u, v, depth = cam.project(pick[1])
            if depth > 0:
                rad = max(4.0, 0.5 * cam.pixels_across(pick[1], 2 * BALL.radius))
                vf = visible_fraction(world.model, world.data, cam, pick[1],
                                      BALL.radius, world.ball_body)
                col = (60, 255, 90) if vf >= 0.5 else (255, 60, 60)
                draw.ellipse([u - rad - 3, v - rad - 3, u + rad + 3, v + rad + 3],
                             outline=col, width=2)
            text = label(cfg) if tag is None else tag(i, cfg)
            draw.rectangle([0, 0, w, 16], fill=(0, 0, 0))
            draw.text((4, 3), text, fill=(255, 255, 255))
            tiles.append(np.asarray(img))

    rows = int(np.ceil(len(tiles) / cols))
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r0, c0 = divmod(i, cols)
        sheet[r0 * h:(r0 + 1) * h, c0 * w:(c0 + 1) * w] = t
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, sheet)
    return path


def rally_video(rig: CameraRig, seeds, path: Path, fps: int = 30):
    """Both recommended feeds, side by side, through a few real feeds."""
    world = World()
    m, d = world.model, world.data
    a, b = camera_pose(rig.a, TABLE, "a"), camera_pose(rig.b, TABLE, "b")
    w, h = rig.a.resolution
    per = max(1, round(1.0 / (fps * m.opt.timestep)))
    frames = []
    with mujoco.Renderer(m, h, w) as r:
        for seed in seeds:
            feed = sample_launch(CFG.launch, seed, TABLE.length)
            mujoco.mj_resetData(m, d)
            d.qpos[world.qpos_a] = READY
            d.qpos[world.qpos_b] = READY
            d.qpos[world.qadr:world.qadr + 3] = feed.pos
            d.qpos[world.qadr + 3:world.qadr + 7] = (1, 0, 0, 0)
            d.qvel[world.vadr:world.vadr + 3] = feed.vel
            mujoco.mj_forward(m, d)
            for _ in range(int(1.8 * fps)):
                world.track()
                for _ in range(per):
                    d.xfrc_applied[world.ball_body, :3] = drag_force(
                        d.qvel[world.vadr:world.vadr + 3], BALL.drag_k)
                    mujoco.mj_step(m, d)
                pair = []
                for cam, name in ((a, "cam_a"), (b, "cam_b")):
                    r.update_scene(d, camera=name)
                    img = Image.fromarray(r.render())
                    dr = ImageDraw.Draw(img)
                    u, v, depth = cam.project(world.ball())
                    if depth > 0:
                        vf = visible_fraction(m, d, cam, world.ball(), BALL.radius,
                                              world.ball_body)
                        rad = max(4.0, 0.5 * cam.pixels_across(world.ball(), 2 * BALL.radius))
                        dr.ellipse([u - rad - 3, v - rad - 3, u + rad + 3, v + rad + 3],
                                   outline=(60, 255, 90) if vf >= 0.5 else (255, 60, 60), width=2)
                    dr.rectangle([0, 0, w, 16], fill=(0, 0, 0))
                    dr.text((4, 3), f"{name}  seed {seed}", fill=(255, 255, 255))
                    pair.append(np.asarray(img))
                frames.append(np.concatenate(pair, axis=1))
            d.xfrc_applied[:] = 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    return path


# -- main ---------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10, help="launcher seeds to roll out")
    p.add_argument("--stride", type=int, default=2, help="keep every Nth ball sample")
    p.add_argument("--top", type=int, default=9, help="candidates in the montage")
    p.add_argument("--noise-px", type=float, default=0.5, help="bearing noise, pixels")
    p.add_argument("--out", type=Path, default=Path("out"))
    p.add_argument("--video", action="store_true")
    args = p.parse_args(argv)

    world = World()
    dt = 0.02 * args.stride
    samples = rollouts(world, range(args.seeds))[:: args.stride]
    print(f"{len(samples)} ball samples from {args.seeds} feeds, both arms tracking\n")

    cands = [
        CameraConfig(height=h, back=bk, pitch_deg=pt, lateral=lat, yaw_deg=yw)
        for h in HEIGHTS for bk in BACKS for pt in PITCHES for lat in LATERALS for yw in YAWS
    ]
    results = []
    for i, c in enumerate(cands):
        results.append(score(world, samples, c, args.noise_px, dt))
        if (i + 1) % 25 == 0:
            print(f"  scored {i + 1}/{len(cands)}")

    # A candidate is only admissible if both cameras hold the whole table at the
    # lens we intend to buy, and the ball is big enough to centroid at all.
    ok = [r for r in results if r["corners"] and r["px"] >= 5.0]
    ranked = sorted(ok, key=RANK)
    worst = sorted(results, key=lambda r: r["both"])[:5]

    lines = [f"ball samples: {len(samples)} from {args.seeds} feeds",
             f"lens: {cands[0].focal_mm} mm on {cands[0].sensor_mm} mm, "
             f"{cands[0].fov_h_deg:.1f} x {cands[0].fov_v_deg:.1f} deg, "
             f"{cands[0].resolution} px, bearing noise {args.noise_px} px", "",
             f"{len(ok)}/{len(results)} candidates hold all four corners at "
             f"{cands[0].focal_mm} mm", "", "-- best --"]
    lines += [row(r) for r in ranked[:15]]
    lines += ["", "-- worst, for contrast --"] + [row(r) for r in worst]

    # Slices through the sweep, because the shape of the answer matters more
    # than its winner: how high is high enough, and how much lateral offset the
    # lens can pay for.
    def best_where(pred):
        got = [r for r in results if pred(r)]
        got = [r for r in got if r["corners"] and r["px"] >= 5.0] or got
        return sorted(got, key=RANK)[0]

    lines += ["", "-- best at each mounting height --"]
    lines += [row(best_where(lambda r, h=h: r["cfg"].height == h)) for h in HEIGHTS]
    lines += ["", "-- best at each distance behind the arm --"]
    lines += [row(best_where(lambda r, b=b: r["cfg"].back == b)) for b in BACKS]
    lines += ["", "-- best at each lateral offset --"]
    lines += [row(best_where(lambda r, l=l: r["cfg"].lateral == l)) for l in LATERALS]
    lines += ["", "-- best at each yaw --"]
    lines += [row(best_where(lambda r, y=y: r["cfg"].yaw_deg == y)) for y in YAWS]
    lines += ["", "-- best at each downward pitch --"]
    lines += [row(best_where(lambda r, p=p: r["cfg"].pitch_deg == p)) for p in PITCHES]
    lines += ["", "-- worst-case self-occlusion, by height (own half, own arm) --"]
    for h in HEIGHTS:
        at = [r for r in results if r["cfg"].height == h]
        lines.append(f"  h{h:.2f}: own-half visibility best {max(r['own_a'] for r in at):.2f} "
                     f"worst {min(r['own_a'] for r in at):.2f}")

    # How much of the table each focal length can hold, at the winning pose.
    best = ranked[0]["cfg"]
    lines += ["", f"-- lens sweep at the recommended pose ({label(best)}) --"]
    for f in FOCALS:
        r = score(world, samples, replace(best, focal_mm=f), args.noise_px, dt)
        lines.append(row(r))

    # The degenerate rig the plan has to avoid, measured rather than asserted.
    flat = replace(best, lateral=0.0, pitch_deg=5.0, height=0.10)
    lines += ["", "-- the degenerate rig: both cameras low, central, level --",
              row(score(world, samples, flat, args.noise_px, dt))]

    text = "\n".join(lines)
    print("\n" + text)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "camera_study.txt").write_text(text + "\n")

    shortlist = [r["cfg"] for r in ranked[:args.top]] or [c for c in cands[:args.top]]
    png = montage(shortlist, samples, args.out / "camera_study.png")
    print(f"\nwrote {png}")
    print(f"wrote {args.out / 'camera_study.txt'}")

    if args.video:
        mp4 = rally_video(CameraRig.symmetric(best), range(4), args.out / "camera_views.mp4")
        print(f"wrote {mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
