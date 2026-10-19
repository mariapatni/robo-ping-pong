"""Make the real arm do what the policy does in sim.

    .venv/bin/python -m hw.mirror --run v6 --standin          # no hardware: a weak sim arm follows
    .venv/bin/python -m hw.mirror --run v6 --speed 0.25       # the real arm, at quarter speed
    .venv/bin/python -m hw.mirror --run v6 --speed 1.0 --seconds 20

One episode of the keep-up policy is run in sim first, recording the joint
targets it sends its actuators - after TableGuard, so exactly what the sim arm
was told - at the control rate. Those targets are then checked (joint range,
table clearance, a speed cap for the real arm) and streamed to the arm at
`--speed` times real time, while its measured angles are logged.

Out come, in out/hw/mirror_<stamp>/:
  log.npz       time, commanded and measured angles, sim angles, ball
  report.txt    per-joint tracking error and lag
  mirror.mp4    sim policy | measured pose drawn in the sim | webcam (real arm)

The measured pose drawn through the same model is the calibration check: if
the middle panel does not look like the webcam, the joint frames are wrong.

`--standin` replaces the real arm with a second MuJoCo arm whose actuators are
weakened to roughly what the 5 V STS3215s deliver (see StandIn), which shows
how much of the policy's motion a weak arm can follow before any hardware
moves.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sim.keepup import KeepUpConfig, KeepUpEnv  # noqa: E402

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


# -- the sim side --------------------------------------------------------------


def record(run: str, seconds: float, seed: int, randomize: bool, ckpt: str | None = None) -> dict:
    """One policy episode: targets, sim joint angles and ball, per control step."""
    from eval_keepup import load_policy

    over = {"episode_s": seconds}
    env = KeepUpEnv(KeepUpConfig.randomized(**over) if randomize else KeepUpConfig(**over))
    act, label = load_policy(run, ckpt)
    obs, _ = env.reset(seed=seed)
    t, target, qsim, ball = [], [], [], []
    done, info = False, {}
    while not done:
        obs, _, term, trunc, info = env.step(act(obs))
        done = term or trunc
        t.append(env.t * env.dt)
        target.append(env.target.copy())
        qsim.append(env.world.qpos)
        ball.append(env.world.ball_pos())
    return {"t": np.array(t), "target": np.array(target), "qsim": np.array(qsim), "ball": np.array(ball),
            "dt": env.dt, "label": label, "bounces": info.get("bounces", 0),
            "cause": info.get("cause", "timeout"), "q0": env.q0}


def speed_limit(target: np.ndarray, dt: float, vmax: float) -> tuple[np.ndarray, float]:
    """Rate-limit a target trajectory to `vmax` rad/s per joint. Returns it and
    the fraction of steps where the limit bit."""
    out = target.copy()
    bit = 0
    for k in range(1, len(out)):
        step = np.clip(out[k] - out[k - 1], -vmax * dt, vmax * dt)
        bit += bool(np.any(np.abs(target[k] - out[k - 1]) > vmax * dt + 1e-9))
        out[k] = out[k - 1] + step
    return out, bit / max(1, len(out) - 1)


# -- the arm side: anything with q(), stream(q), and start()/stop() ---------------


class StandIn:
    """A second MuJoCo arm, weakened toward the real one.

    The sim's actuator is a 998 N.m/rad position servo saturating at 2.94 N.m,
    i.e. full torque at 0.17 deg of error. Measured on the real arm with P=32:
    output rises ~0.8% per tick of error, so full output needs ~0.18 rad -
    roughly 1/170th of the sim's stiffness - and the 5 V supply leaves
    perhaps half the rated torque. Both numbers are estimates until the arm
    is rehomed and identified properly; they are arguments, not facts.
    """

    def __init__(self, torque: float = 1.0, full_at: float = 0.18, lag_s: float = 0.0):
        from sim.env import EnvConfig, PingPongEnv
        from sim.paddle import PaddleConfig

        self.world = PingPongEnv(EnvConfig(paddle=PaddleConfig()))
        m = self.world.model
        kp = torque / full_at
        for a in self.world.actuator_ids:
            scale = kp / m.actuator_gainprm[a, 0]  # damping scaled with stiffness
            m.actuator_gainprm[a, 0] = kp
            m.actuator_biasprm[a, 1] = -kp
            m.actuator_biasprm[a, 2] *= scale
            m.actuator_forcerange[a] = (-torque, torque)
        self.world.set_ball((2.0, 2.0, -0.5))
        self.t = 0.0

    def start(self, q0) -> None:
        self.world.reset(np.asarray(q0, float))
        self.world.set_ball((2.0, 2.0, -0.5))

    def q(self) -> np.ndarray:
        return self.world.qpos

    def stream(self, q, dt: float) -> None:
        self.world.command(q)
        n = max(1, round(dt / self.world.model.opt.timestep))
        self.world.step(n)

    def stop(self) -> None:
        pass


class Real:
    def __init__(self, torque_cap: float = 0.6):
        from .so101 import SO101, load_calibration

        if load_calibration() is None:
            raise SystemExit("refusing: the arm has not been rehomed (hw/calibration.json missing), "
                             "so sim angles do not mean the same thing on it. Run `python -m hw.rehome`.")
        self.arm = SO101()
        self.torque_cap = torque_cap

    def start(self, q0) -> None:
        a = self.arm
        a.configure(acceleration=0, torque_limit=int(self.torque_cap * 1000))
        a.torque(True)
        a.unfold()
        a.move_to(q0, speed=0.25)

    def q(self) -> np.ndarray:
        return self.arm.q()

    def stream(self, q, dt: float) -> None:
        # The sim already kept these off the table; the guard is the same
        # one, so this only refuses when the calibration has drifted.
        if self.arm.check.guard.is_safe(q):
            self.arm.write_ticks(self.arm.to_ticks(q))

    def stop(self) -> None:
        a = self.arm
        a.hold()
        a.configure(acceleration=40, torque_limit=int(self.torque_cap * 1000))
        a.fold()
        a.torque(False)


# -- playback and analysis -----------------------------------------------------------


def play(arm, target: np.ndarray, dt: float, speed: float, realtime: bool) -> dict:
    """Stream targets every dt/speed seconds; log what was asked and what was measured."""
    period = dt / speed
    cmd, meas, stamps = [], [], []
    t0 = time.perf_counter()
    for k, q in enumerate(target):
        arm.stream(q, period)
        cmd.append(q)
        meas.append(arm.q())
        stamps.append(time.perf_counter() - t0)
        if realtime:
            time.sleep(max(0.0, t0 + (k + 1) * period - time.perf_counter()))
    return {"cmd": np.array(cmd), "meas": np.array(meas), "wall": np.array(stamps)}


def lag_steps(cmd: np.ndarray, meas: np.ndarray, max_lag: int = 30) -> int:
    """Shift (in steps) that best lines the measured motion up with the command."""
    if len(cmd) < 3 * max_lag or np.allclose(np.diff(cmd), 0):
        return 0
    c, m = np.diff(cmd), np.diff(meas)
    errs = [np.mean((c[: len(c) - s] - m[s:]) ** 2) for s in range(max_lag)]
    return int(np.argmin(errs))


def report(rec: dict, log: dict, speed: float, vmax: float, limited: float) -> str:
    cmd, meas = log["cmd"], log["meas"]
    err = meas - cmd
    lines = [f"{rec['label']}: {len(cmd)} steps at {1 / rec['dt']:.0f} Hz sim, played at {speed}x "
             f"({len(cmd) * rec['dt'] / speed:.1f} s wall); sim episode: {rec['bounces']} bounces, "
             f"ended by {rec['cause']}",
             f"real-arm speed cap {vmax:.2f} rad/s in sim time ({vmax * speed:.2f} rad/s wall) "
             f"bit on {limited:.0%} of steps",
             "",
             f"{'joint':14s} {'cmd range':>16s} {'peak cmd v':>11s} {'rms err':>8s} {'max err':>8s} {'lag':>9s}"]
    for j, name in enumerate(JOINTS):
        v = np.abs(np.diff(cmd[:, j])).max() / rec["dt"] if len(cmd) > 1 else 0.0
        lag = lag_steps(cmd[:, j], meas[:, j]) * rec["dt"] / speed
        lines.append(f"{name:14s} {cmd[:, j].min():+6.2f}..{cmd[:, j].max():+6.2f} rad "
                     f"{v:8.2f}/s {np.degrees(np.sqrt(np.mean(err[:, j] ** 2))):6.1f} deg "
                     f"{np.degrees(np.abs(err[:, j]).max()):6.1f} deg {lag * 1000:6.0f} ms")
    return "\n".join(lines)


def video(path: Path, rec: dict, log: dict, cam_frames: list | None, size=(480, 400)) -> None:
    """Sim policy | measured pose through the same model | webcam."""
    import imageio.v2 as imageio
    import mujoco

    from sim.env import EnvConfig, PingPongEnv
    from sim.paddle import PaddleConfig

    w, h = size
    world = PingPongEnv(EnvConfig(paddle=PaddleConfig()), offscreen=(w, h))
    r = mujoco.Renderer(world.model, h, w)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.15, 0.0, 0.15)
    cam.distance, cam.azimuth, cam.elevation = 1.0, 135.0, -15.0

    def draw(q, ball):
        world.reset(q)
        world.set_ball(ball)
        mujoco.mj_forward(world.model, world.data)
        r.update_scene(world.data, cam)
        return r.render()

    fps = int(round(1 / rec["dt"]))
    with imageio.get_writer(path, fps=fps, macro_block_size=8) as out:
        for k in range(len(log["cmd"])):
            panels = [draw(rec["qsim"][k], rec["ball"][k]), draw(log["meas"][k], (2.0, 2.0, -0.5))]
            if cam_frames is not None:
                import cv2

                f = cam_frames[min(k, len(cam_frames) - 1)]
                panels.append(cv2.resize(f, (w, h))[:, :, ::-1] if f is not None else np.zeros((h, w, 3), np.uint8))
            out.append_data(np.hstack(panels))


class Camera:
    """Grabs the newest webcam still the tmux `cam` window keeps writing."""

    def __init__(self, path: Path = ROOT / "out" / "hw" / "latest.jpg"):
        self.path = path
        self.frames = []

    def grab(self) -> None:
        import cv2

        self.frames.append(cv2.imread(str(self.path)))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="v6")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--seconds", type=float, default=10.0, help="length of the sim episode")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--speed", type=float, default=0.25, help="playback speed, 1.0 = real time")
    p.add_argument("--vmax", type=float, default=4.5, help="per-joint speed cap, rad/s of sim time")
    p.add_argument("--standin", action="store_true", help="no hardware: a weakened sim arm follows")
    p.add_argument("--standin-torque", type=float, default=1.0, help="N.m")
    p.add_argument("--torque-cap", type=float, default=0.6, help="real arm: fraction of full servo output")
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args(argv)

    out = ROOT / "out" / "hw" / f"mirror_{time.strftime('%Y%m%d_%H%M%S')}{'_standin' if args.standin else ''}"
    out.mkdir(parents=True, exist_ok=True)

    rec = record(args.run, args.seconds, args.seed, args.randomize, args.ckpt)
    print(f"{rec['label']}: {len(rec['target'])} steps, {rec['bounces']} bounces, ended by {rec['cause']}", flush=True)
    target, limited = speed_limit(rec["target"], rec["dt"], args.vmax)

    arm = StandIn(torque=args.standin_torque) if args.standin else Real(args.torque_cap)
    camera = None if args.standin else Camera()
    arm.start(target[0])
    try:
        if camera is None:
            log = play(arm, target, rec["dt"], args.speed, realtime=False)
        else:
            # Grab a webcam still alongside every step (the cam window writes
            # them at 2 Hz, so at low speeds consecutive steps share one).
            class Logged:
                def __init__(s, a): s.a = a
                def q(s): camera.grab(); return s.a.q()  # noqa: E704
                def stream(s, q, dt): s.a.stream(q, dt)  # noqa: E704
            log = play(Logged(arm), target, rec["dt"], args.speed, realtime=True)
    finally:
        arm.stop()

    text = report(rec, log, args.speed, args.vmax, limited)
    print(text, flush=True)
    (out / "report.txt").write_text(text + "\n")
    np.savez(out / "log.npz", t=rec["t"], target=rec["target"], cmd=log["cmd"], meas=log["meas"],
             wall=log["wall"], qsim=rec["qsim"], ball=rec["ball"])
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    if not args.no_video:
        video(out / "mirror.mp4", rec, log, camera.frames if camera else None)
        print("wrote", out / "mirror.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
