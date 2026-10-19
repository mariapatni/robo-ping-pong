"""How much of the keep-up policy survives a weaker arm than it trained on.

    .venv/bin/python -m hw.weak_arm --run v6 --episodes 20

The sim's STS3215 is a 998 N.m/rad position servo saturating at 2.94 N.m:
full torque at 0.17 deg of error. The real arm, at P=32 on 5 V, reaches full
output only around 0.18 rad of error and has an unmeasured fraction of the
rated torque. This scores the policy - closed loop, seeing the ball - with
the actuators set to a grid of (torque, error-at-full-torque) between the two.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sim.keepup import KeepUpConfig, KeepUpEnv  # noqa: E402


def weaken(env: KeepUpEnv, torque: float, full_at: float) -> None:
    m = env.world.model
    for a in env.world.actuator_ids:
        kp = torque / full_at
        scale = kp / m.actuator_gainprm[a, 0]
        m.actuator_gainprm[a, 0] = kp
        m.actuator_biasprm[a, 1] = -kp
        m.actuator_biasprm[a, 2] *= scale
        m.actuator_forcerange[a] = (-torque, torque)


def score(act, torque, full_at, episodes, seconds, seed0=20_000):
    env = KeepUpEnv(KeepUpConfig(episode_s=seconds))
    if torque is not None:
        weaken(env, torque, full_at)
    b, dur = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(act(obs))
            done = term or trunc
        b.append(info["bounces"])
        dur.append(env.t * env.dt)
    return np.mean(b), np.median(dur), np.mean(np.array(dur) >= seconds - 2 * env.dt)


def main(argv=None) -> int:
    from eval_keepup import load_policy

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="v6")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seconds", type=float, default=20.0)
    args = p.parse_args(argv)
    act, label = load_policy(args.run, None)
    print(f"{label}, {args.episodes} centre drops x {args.seconds:.0f} s, fixed world", flush=True)
    print(f"{'torque N.m':>10s} {'full at':>9s} {'kp N.m/rad':>11s} {'bounces':>8s} {'median s':>9s} {'full-length':>11s}")
    grid = [(None, None)] + [(t, f) for t in (2.94, 1.5, 1.0, 0.6) for f in (0.02, 0.06, 0.18)]
    for t, f in grid:
        b, d, full = score(act, t, f, args.episodes, args.seconds)
        kp = "998 (sim)" if t is None else f"{t / f:.0f}"
        print(f"{'2.94 sim' if t is None else t:>10} {'0.003' if f is None else f:>9} {kp:>11s} "
              f"{b:8.1f} {d:9.1f} {full:11.0%}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
