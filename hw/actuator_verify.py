"""How the real servos answer a command, how noisy that is, and how far the sim is from it.

    .venv/bin/python -m hw.actuator_verify --dry                # plan, safety check, sim prediction; arm untouched
    .venv/bin/python -m hw.actuator_verify                      # run it on the arm (~6 min, torque capped at 60%)
    .venv/bin/python -m hw.actuator_verify --analyse out/hw/actuator_verify_<stamp>   # re-analyse a recording

The arm is driven from the keep-up start pose through four programs, every
command and every position reading stamped on one clock:

  steps    each joint stepped +/-0.05, 0.10, 0.25 rad and back, `--repeats` times
  chirp    each joint swept as a 0.05 rad sine from 0.5 to 5 Hz, at the 30 Hz command rate
  policy   a real 30 Hz command stream recorded from the keep-up policy in sim, `--repeats` times
  rest     held still, for the encoders' own noise

From those it reports, per joint:

  - dead time (command to first 5% of the move), time to 90%, mid-move speed,
    overshoot and settled error - each as a mean and a spread over repeats,
    which is the noise on a single command;
  - lag and RMS error when tracking the chirp and the policy stream;
  - encoder noise at rest;
  - a fitted actuator model: the command is delayed by `delay`, rate-limited
    to `vmax`, and followed with time constant `tau`, plus a steady offset;
  - the VERIFICATION: the identical command sequences are replayed through the
    MuJoCo arm (as v1-v7 trained on it) and through the fitted model, and each
    is scored by its RMS distance from what the real arm did. The fitted
    parameters, with their spread, are what the sim's actuators should be
    randomised over. A future sim with that model built in should score close
    to the fitted model's number here.

Out: out/hw/actuator_verify_<stamp>/{raw.npz, segments.json, report.txt, fit.json}.

Safety, as in hw/joint_test.py: torque capped (`--torque-cap`, default 0.6),
every target checked against the joint limits and the sim's table guard before
anything moves, a load watchdog, and the arm folded and released at the end -
including on Ctrl-C.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRID_DT = 0.002  # s, the uniform grid tracking errors and model fits are computed on


# -- the programs ----------------------------------------------------------------


@dataclass
class Segment:
    """One continuous recording: commands at times `t`, then `tail` s of watching."""

    name: str
    program: str
    joint: int | None  # the joint being exercised, or None for all
    t: np.ndarray  # command times, s from segment start
    q: np.ndarray  # (len(t), 5) joint targets, rad
    tail: float = 0.8
    meta: dict = field(default_factory=dict)


def step_segments(q0, joints, amps=(0.05, 0.10, 0.25), repeats=5, hold=0.8) -> list[Segment]:
    """Per joint and amplitude: +a, back, -a, back, repeated. One segment per
    joint and amplitude, so a repeat is directly comparable to the others."""
    segs = []
    for j in joints:
        for a in amps:
            t, q = [], []
            for r in range(repeats):
                for sgn in (+1, -1):
                    for target in (a * sgn, 0.0):
                        t.append(len(t) * hold)
                        qq = np.array(q0, float)
                        qq[j] += target
                        q.append(qq)
            segs.append(Segment(f"steps_{JOINTS[j]}_{a:.2f}", "steps", j, np.array(t), np.array(q),
                                tail=hold, meta={"amp": a, "hold": hold, "repeats": repeats}))
    return segs


def chirp_segments(q0, joints, amp=0.05, f0=0.5, f1=5.0, seconds=8.0, rate=30.0) -> list[Segment]:
    """A sine whose frequency rises linearly, sent as the policy sends commands:
    a new target every 1/rate s, held in between."""
    segs = []
    t = np.arange(0.0, seconds, 1.0 / rate)
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / seconds * t * t)
    for j in joints:
        q = np.tile(np.asarray(q0, float), (len(t), 1))
        q[:, j] += amp * np.sin(phase)
        segs.append(Segment(f"chirp_{JOINTS[j]}", "chirp", j, t, q, meta={"amp": amp, "f0": f0, "f1": f1}))
    return segs


def policy_segments(run="v7", ckpt="54009728", seconds=10.0, repeats=3, seed=30033) -> list[Segment]:
    """The keep-up policy's own 30 Hz targets from one sim episode - exactly what
    the robot would be sent - played `repeats` times."""
    from hw.mirror import record

    rec = record(run, seconds, seed, randomize=True, ckpt=ckpt)
    t = rec["t"] - rec["t"][0]
    return [Segment(f"policy_{run}_{r}", "policy", None, t, rec["target"], meta={"run": run, "ckpt": ckpt,
                    "seed": seed}) for r in range(repeats)]


def rest_segment(q0, seconds=3.0) -> Segment:
    return Segment("rest", "rest", None, np.array([0.0]), np.array([q0], float), tail=seconds)


def build_programs(q0, programs, joints, repeats) -> list[Segment]:
    segs = []
    if "rest" in programs:
        segs.append(rest_segment(q0))
    if "steps" in programs:
        segs += step_segments(q0, joints, repeats=repeats)
    if "chirp" in programs:
        segs += chirp_segments(q0, joints)
    if "policy" in programs:
        segs += policy_segments(repeats=max(1, min(repeats, 3)))
    return segs


def unsafe_targets(segs, lower, upper, guard) -> list[str]:
    """Every segment with a target outside the joint limits or near the table."""
    bad = []
    for s in segs:
        if np.any(s.q < lower - 1e-9) or np.any(s.q > upper + 1e-9):
            bad.append(f"{s.name}: outside the joint limits")
        elif not all(guard.is_safe(q) for q in np.unique(np.round(s.q, 4), axis=0)):
            bad.append(f"{s.name}: too near the table in sim")
    return bad


# -- what a recording is ---------------------------------------------------------


@dataclass
class Recording:
    """Commands as sent and positions as read, on one clock (s from segment start)."""

    cmd_t: np.ndarray
    cmd_q: np.ndarray
    read_t: np.ndarray
    read_q: np.ndarray

    def commanded_at(self, t) -> np.ndarray:
        """Zero-order hold: the latest command sent at or before each time."""
        k = np.searchsorted(self.cmd_t, t, side="right") - 1
        return self.cmd_q[np.clip(k, 0, len(self.cmd_q) - 1)]

    def measured_at(self, t) -> np.ndarray:
        return np.column_stack([np.interp(t, self.read_t, self.read_q[:, j]) for j in range(5)])

    def grid(self) -> np.ndarray:
        return np.arange(max(self.read_t[0], self.cmd_t[0]), self.read_t[-1], GRID_DT)


# -- analysis --------------------------------------------------------------------


def step_metrics(rec: Recording, j: int) -> list[dict]:
    """Delay, rise, speed, overshoot and settling for every command that moved joint j."""
    out = []
    for k in range(1, len(rec.cmd_t)):
        move = rec.cmd_q[k, j] - rec.cmd_q[k - 1, j]
        if abs(move) < 1e-3:
            continue
        t0 = rec.cmd_t[k]
        t1 = rec.cmd_t[k + 1] if k + 1 < len(rec.cmd_t) else rec.read_t[-1]
        m = (rec.read_t >= t0) & (rec.read_t < t1)
        if m.sum() < 10:
            continue
        tt, y = rec.read_t[m] - t0, rec.read_q[m, j]
        start = np.interp(t0, rec.read_t, rec.read_q[:, j])
        frac = (y - start) / move

        def first(level):
            i = np.argmax(frac >= level)
            return float(tt[i]) if frac[i] >= level else np.nan

        t05, t20, t80, t90 = first(0.05), first(0.2), first(0.8), first(0.9)
        tail = y[tt > tt[-1] - 0.2]
        out.append({"move": float(move), "dead_ms": 1000 * t05, "t90_ms": 1000 * t90,
                    "speed": float(0.6 * abs(move) / (t80 - t20)) if t80 > t20 else np.nan,
                    "overshoot_pct": float(max(0.0, (frac.max() - 1.0) * 100)),
                    "settled_err_deg": float(np.degrees(tail.mean() - rec.cmd_q[k, j]))})
    return out


def lag_and_rms(rec: Recording, j: int, max_lag=0.4) -> tuple[float, float]:
    """Lag (s) maximising the cross-correlation of commanded and measured motion,
    and the RMS error (rad) once that lag is removed."""
    g = rec.grid()
    u, y = rec.commanded_at(g)[:, j], rec.measured_at(g)[:, j]
    u, y = u - u.mean(), y - y.mean()
    lags = np.arange(0, int(max_lag / GRID_DT))
    scores = [np.dot(u[: len(u) - L], y[L:]) for L in lags]
    L = int(lags[int(np.argmax(scores))])
    return L * GRID_DT, float(np.sqrt(np.mean((u[: len(u) - L] - y[L:]) ** 2)))


def simulate_model(rec: Recording, j: int, delay, tau, vmax, bias=0.0) -> np.ndarray:
    """The simple actuator model on the recording's commands, vectorised over
    parameter arrays: the command is delayed by `delay`, and the position
    follows it with time constant `tau`, never faster than `vmax`. Returns
    (len(grid), n_params) positions, starting from the measured position."""
    g = rec.grid()
    delay, tau, vmax = (np.atleast_1d(np.asarray(v, float)) for v in (delay, tau, vmax))
    y = np.full(np.broadcast(delay, tau, vmax).shape, rec.measured_at(g[:1])[0, j])
    cmd = np.stack([rec.commanded_at(g - d)[:, j] for d in np.unique(delay)])
    which = np.searchsorted(np.unique(delay), np.broadcast_to(delay, y.shape))
    out = np.empty((len(g), y.size))
    alpha = np.broadcast_to(GRID_DT / np.maximum(tau, GRID_DT), y.shape)
    vstep = np.broadcast_to(vmax * GRID_DT, y.shape)
    for i in range(len(g)):
        target = cmd[which, i] + bias
        y = y + np.clip((target - y) * alpha, -vstep, vstep)
        out[i] = y
    return out


def fit_model(recs: list[Recording], j: int) -> dict:
    """Grid-search delay, tau and vmax to the recordings; then the offset."""
    delays = np.arange(0.0, 0.161, 0.01)
    taus = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15])
    vmaxes = np.geomspace(0.3, 12.0, 16)
    D, T, V = (a.ravel() for a in np.meshgrid(delays, taus, vmaxes, indexing="ij"))
    sse = np.zeros(D.size)
    for rec in recs:
        y = rec.measured_at(rec.grid())[:, j]
        sse += ((simulate_model(rec, j, D, T, V) - y[:, None]) ** 2).sum(axis=0)
    i = int(np.argmin(sse))
    p = {"delay_s": float(D[i]), "tau_s": float(T[i]), "vmax": float(V[i])}
    resid = np.concatenate([rec.measured_at(rec.grid())[:, j] - simulate_model(rec, j, D[i], T[i], V[i])[:, 0]
                            for rec in recs])
    p["bias_deg"] = float(np.degrees(resid.mean()))
    p["rms_deg"] = float(np.degrees(np.sqrt(np.mean((resid - resid.mean()) ** 2))))
    return p


def replay_in_sim(rec: Recording, world=None) -> np.ndarray:
    """The same commands through the MuJoCo arm v1-v7 trained on. Returns joint
    angles on the recording's grid."""
    import mujoco

    if world is None:
        from sim.config import EnvConfig
        from sim.env import PingPongEnv
        from sim.paddle import PaddleConfig

        world = PingPongEnv(EnvConfig(paddle=PaddleConfig()))
    g = rec.grid()
    world.reset(rec.measured_at(g[:1])[0])
    m, d = world.model, world.data
    t, out, i = g[0], np.empty((len(g), 5)), 0
    while i < len(g):
        world.command(rec.commanded_at(np.array([t]))[0])
        mujoco.mj_step(m, d)
        t += m.opt.timestep
        while i < len(g) and g[i] <= t:
            out[i] = world.qpos
            i += 1
    return out


def summarise(segs: list[Segment], recs: list[Recording], world=None) -> tuple[str, dict]:
    """The report and the fitted parameters."""
    lines, fit = [], {}
    by = lambda prog: [(s, r) for s, r in zip(segs, recs) if s.program == prog]  # noqa: E731

    rest = by("rest")
    if rest:
        q = rest[0][1].read_q
        lines += ["ENCODER NOISE AT REST (std)", "  " + "  ".join(
            f"{JOINTS[j]} {np.degrees(q[:, j].std()):.3f} deg" for j in range(5)), ""]

    steps = by("steps")
    if steps:
        lines += ["STEP RESPONSE: mean +/- std over repeats (the std is the noise on one command)",
                  f"  {'joint':14s} {'step':>5s} {'dead time':>14s} {'t90':>14s} {'speed rad/s':>12s} "
                  f"{'overshoot':>10s} {'settled':>12s}"]
        for s, r in steps:
            ms = step_metrics(r, s.joint)
            # Steps of this size, out and back: commands are rounded to whole
            # encoder ticks (0.0015 rad), so the size is matched loosely.
            ms = [x for x in ms if abs(abs(x["move"]) - s.meta["amp"]) < 0.004]
            if not ms:
                continue
            col = lambda k: np.array([x[k] for x in ms], float)  # noqa: E731
            f = lambda k, w=0: f"{np.nanmean(col(k)):6.{w}f} +/- {np.nanstd(col(k)):<5.{w}f}"  # noqa: E731
            lines.append(f"  {JOINTS[s.joint]:14s} {s.meta['amp']:5.2f} {f('dead_ms')}ms {f('t90_ms')}ms "
                         f"{f('speed', 2)} {np.nanmean(col('overshoot_pct')):8.1f}% {f('settled_err_deg', 2)}deg")
        lines.append("")

    track = by("chirp") + by("policy")
    if track:
        lines += ["TRACKING (lag that best aligns command and motion; RMS error after removing it)"]
        for s, r in track:
            js = [s.joint] if s.joint is not None else range(5)
            cells = []
            for j in js:
                lag, rms = lag_and_rms(r, j)
                cells.append(f"{JOINTS[j]} {1000 * lag:.0f}ms/{np.degrees(rms):.1f}deg")
            lines.append(f"  {s.name:24s} " + "  ".join(cells))
        lines.append("")

    fit_recs = {j: [r for s, r in steps + track if s.joint in (j, None)] for j in range(5)}
    lines += ["FITTED ACTUATOR MODEL and VERIFICATION (RMS distance from the real arm over every recording)",
              f"  {'joint':14s} {'delay':>7s} {'tau':>7s} {'vmax':>8s} {'offset':>8s} "
              f"{'fitted model':>13s} {'current sim':>12s}"]
    for j in range(5):
        if not fit_recs[j]:
            continue
        p = fit_model(fit_recs[j], j)
        sim_err = np.concatenate([replay_in_sim(r, world)[:, j] - r.measured_at(r.grid())[:, j]
                                  for r in fit_recs[j]])
        p["sim_rms_deg"] = float(np.degrees(np.sqrt(np.mean(sim_err ** 2))))
        # Spread: refit on each recording alone - what the sim should randomise over.
        if len(fit_recs[j]) > 1:
            each = [fit_model([r], j) for r in fit_recs[j]]
            p["spread"] = {k: [float(min(e[k] for e in each)), float(max(e[k] for e in each))]
                           for k in ("delay_s", "tau_s", "vmax")}
        fit[JOINTS[j]] = p
        lines.append(f"  {JOINTS[j]:14s} {1000 * p['delay_s']:5.0f}ms {1000 * p['tau_s']:5.0f}ms "
                     f"{p['vmax']:6.2f}/s {p['bias_deg']:+6.2f}deg {p['rms_deg']:11.2f}deg "
                     f"{p['sim_rms_deg']:10.2f}deg")
    lines += ["", "A good actuator model in the sim brings 'current sim' down to near 'fitted model'."]
    return "\n".join(lines) + "\n", fit


# -- saving and loading ----------------------------------------------------------


def save(out: Path, segs, recs, args) -> None:
    arrays = {}
    for i, r in enumerate(recs):
        for k in ("cmd_t", "cmd_q", "read_t", "read_q"):
            arrays[f"{i}_{k}"] = getattr(r, k)
    np.savez(out / "raw.npz", **arrays)
    (out / "segments.json").write_text(json.dumps(
        {"args": vars(args), "segments": [{"name": s.name, "program": s.program, "joint": s.joint,
                                           "tail": s.tail, "meta": s.meta} for s in segs]}, indent=2, default=str))


def load(out: Path) -> tuple[list[Segment], list[Recording]]:
    meta = json.loads((out / "segments.json").read_text())["segments"]
    z = np.load(out / "raw.npz")
    recs = [Recording(*(z[f"{i}_{k}"] for k in ("cmd_t", "cmd_q", "read_t", "read_q"))) for i in range(len(meta))]
    segs = [Segment(m["name"], m["program"], m["joint"], r.cmd_t, r.cmd_q, m["tail"], m["meta"])
            for m, r in zip(meta, recs)]
    return segs, recs


# -- the hardware run ------------------------------------------------------------


def run_segment(arm, seg: Segment, torque_cap: float, max_overload_s: float = 0.5) -> Recording:
    """Send the segment's commands on time; read positions as fast as the bus
    allows in between; abort if a joint stays near the torque cap."""
    from hw import feetech as ft

    arm.move_to(seg.q[0], speed=0.3)
    cmd_t, cmd_q, read_t, read_q = [], [], [], []
    t0 = time.perf_counter()
    k, n, end = 0, len(seg.t), seg.t[-1] + seg.tail
    last_watch, heavy_since = 0.0, None
    while True:
        now = time.perf_counter() - t0
        if k < n and now >= seg.t[k]:
            ticks = arm.to_ticks(seg.q[k])
            arm.write_ticks(ticks)
            cmd_t.append(time.perf_counter() - t0)
            cmd_q.append(arm.to_rad(ticks))  # what was actually sent, after rounding to ticks
            k += 1
            continue
        if now > end:
            break
        try:
            a = time.perf_counter()
            ticks = arm.ticks()
            b = time.perf_counter()
            read_t.append((a + b) / 2 - t0)
            read_q.append(arm.to_rad(ticks))
        except ft.BusError:
            pass
        if now - last_watch > 0.05:
            last_watch = now
            heavy = np.abs(arm.loads()).max() > 0.9 * torque_cap
            heavy_since = (heavy_since or now) if heavy else None
            if heavy_since is not None and now - heavy_since > max_overload_s:
                arm.hold()
                raise RuntimeError(f"{seg.name}: a joint stayed at the torque cap for {max_overload_s} s - stopped")
    return Recording(np.array(cmd_t), np.array(cmd_q), np.array(read_t), np.array(read_q))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry", action="store_true", help="plan, check and predict in sim only; the arm is not touched")
    p.add_argument("--analyse", type=Path, default=None, help="re-analyse a saved recording directory")
    p.add_argument("--programs", nargs="*", default=["rest", "steps", "chirp", "policy"])
    p.add_argument("--joints", nargs="*", default=list(JOINTS))
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--torque-cap", type=float, default=0.6,
                   help="fraction of full servo output; 1.0 measures the true limits - only with someone watching")
    args = p.parse_args(argv)

    if args.analyse:
        segs, recs = load(args.analyse)
        report, fit = summarise(segs, recs)
        (args.analyse / "report.txt").write_text(report)
        (args.analyse / "fit.json").write_text(json.dumps(fit, indent=2))
        print(report)
        return 0

    from sim.keepup import KeepUpEnv

    env = KeepUpEnv()
    q0, joints = env.q0, [JOINTS.index(j) for j in args.joints]
    segs = build_programs(q0, args.programs, joints, args.repeats)
    bad = unsafe_targets(segs, env.world.lower, env.world.upper, env.guard)
    duration = sum(s.t[-1] + s.tail + 1.5 for s in segs)  # + ~1.5 s to move to each start
    print(f"{len(segs)} segments, about {duration / 60:.1f} min of arm motion")
    if bad:
        print("REFUSED - unsafe targets:\n  " + "\n  ".join(bad))
        return 1

    if args.dry:
        # The sim arm stands in for the real one: commands go through replay_in_sim
        # as if measured, so the whole analysis runs and the plan is exercised.
        recs = []
        for s in segs:
            t = np.arange(0.0, s.t[-1] + s.tail, 0.002)
            fake = Recording(s.t, s.q, t, np.tile(s.q[0], (len(t), 1)))
            recs.append(Recording(s.t, s.q, fake.grid(), replay_in_sim(fake, env.world)))
        report, _ = summarise(segs, recs, env.world)
        print("DRY RUN - the 'real arm' below is the MuJoCo arm itself, so this shows what\n"
              "the current sim predicts. Nothing was sent to the servos.\n\n" + report)
        return 0

    from hw.so101 import SO101

    out = ROOT / "out" / "hw" / f"actuator_verify_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    arm = SO101()
    arm.configure(acceleration=0, torque_limit=int(args.torque_cap * 1000))
    arm.torque(True)
    recs = []
    try:
        arm.unfold()
        arm.move_to(q0, speed=0.25)
        for i, s in enumerate(segs):
            print(f"[{i + 1}/{len(segs)}] {s.name}", flush=True)
            recs.append(run_segment(arm, s, args.torque_cap))
    finally:
        arm.hold()
        arm.configure(acceleration=40, torque_limit=int(args.torque_cap * 1000))
        arm.fold()
        arm.torque(False)
        if recs:
            save(out, segs[: len(recs)], recs, args)
    report, fit = summarise(segs[: len(recs)], recs, arm.check.world)
    (out / "report.txt").write_text(report)
    (out / "fit.json").write_text(json.dumps(fit, indent=2))
    print(report + f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
