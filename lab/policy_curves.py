"""Rebuild each keep-up version's performance over training from its checkpoints.

    .venv/bin/python -m lab.policy_curves                    every 2M steps, 8 drops of up to 120 s
    .venv/bin/python -m lab.policy_curves --runs v5 v6 --every 1000000

v1-v7 were trained without logging, so their loss curves were never written.
The checkpoints were kept, though (every 0.5M steps): this evaluates them on
the same held-out drops and writes policies/keepup/<run>/curve.json - median
and mean time the ball stayed up, against environment steps - which the
dashboard's Training page plots. 120 s drops, because at 30 s most
checkpoints from 4M steps on lasted the whole drop and the curves were flat.
Runs whose inputs no longer match the
environment (v1, v2) are skipped. Progress shows on the dashboard's Live page.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main(argv=None) -> int:
    from eval_keepup import load_policy

    from lab.live import Live
    from sim.keepup import KeepUpConfig, KeepUpEnv

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", nargs="*", default=None)
    p.add_argument("--every", type=int, default=2_000_000)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--episode-s", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=10_000)
    a = p.parse_args(argv)

    base = ROOT / "policies" / "keepup"
    runs = a.runs or sorted((d.name for d in base.glob("v*") if (d / "ckpt").exists()), key=lambda s: int(s[1:]))
    live = Live("policy_curves", "Rebuilding training curves from checkpoints",
                f"Evaluates every {a.every / 1e6:g}M-step checkpoint of {', '.join(runs)} on {a.episodes} held-out "
                f"randomised drops of up to {a.episode_s:.0f} s, and records how long the ball stayed up. "
                "Writes policies/keepup/<run>/curve.json for the Training page.", kind="training")
    env = KeepUpEnv(KeepUpConfig.randomized(episode_s=a.episode_s))
    try:
        for run in runs:
            steps = sorted(int(m.group(1)) for f in (base / run / "ckpt").glob("keepup_*_steps.zip")
                           if (m := re.search(r"keepup_(\d+)_steps", f.name)))
            pick, last = [], -10**12
            for s in steps:
                if s - last >= a.every or s == steps[-1]:
                    pick.append(s)
                    last = s
            points = []
            for k, s in enumerate(pick):
                try:
                    act, _ = load_policy(run, str(s))
                    obs, _ = env.reset(seed=a.seed)
                    act(obs)
                except Exception as e:  # inputs no longer match the environment
                    live.log(f"{run}: skipped ({type(e).__name__}: inputs do not match the environment)")
                    break
                secs, bounces = [], []
                for ep in range(a.episodes):
                    obs, _ = env.reset(seed=a.seed + ep)
                    done = False
                    while not done:
                        obs, _, term, trunc, info = env.step(act(obs))
                        done = term or trunc
                    secs.append(env.t * env.dt)
                    bounces.append(info["bounces"])
                pt = {"steps": s, "median_s": float(np.median(secs)), "mean_s": float(np.mean(secs)),
                      "mean_bounces": float(np.mean(bounces))}
                points.append(pt)
                live.point(f"{run} median s", s / 1e6, pt["median_s"])
                live.status(run=run, checkpoint=f"{k + 1}/{len(pick)}", steps=s, median_s=pt["median_s"])
                (base / run / "curve.json").write_text(json.dumps(
                    {"episodes": a.episodes, "episode_s": a.episode_s, "seed": a.seed, "points": points}, indent=1))
            if points:
                live.log(f"{run}: {len(points)} checkpoints, best median {max(p['median_s'] for p in points):.1f} s")
        live.done()
    except Exception as e:
        live.failed(repr(e))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
