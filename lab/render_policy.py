"""Play a keep-up policy in the sim, live on the dashboard, and save the video.

    .venv/bin/python -m lab.render_policy --run v5 --seed 10000 --episodes 2

Every frame goes to the dashboard's Live page as it is rendered; the finished
video lands in out/dashboard/videos/<run>_seed<seed>.mp4, which the Policies
page lists under that version.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main(argv=None) -> int:
    import cv2
    import imageio.v2 as imageio
    from eval_keepup import load_policy

    from lab.live import Live
    from sim.keepup import KeepUpConfig, KeepUpEnv

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", default="v5")
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--episode-s", type=float, default=60.0)
    a = p.parse_args(argv)

    live = Live(f"policy_{a.run}", f"Policy {a.run}, drop seed {a.seed}",
                f"Keep-up policy {a.run} in the randomised world, {a.episodes} drop(s) from seed {a.seed}, up to "
                f"{a.episode_s:.0f} s each. Each drop: the ball falls from somewhere over the paddle; the policy has "
                "to catch it and keep it bouncing over the centre spot.", kind="policy")
    try:
        env = KeepUpEnv(KeepUpConfig.randomized(episode_s=a.episode_s), render_size=(800, 600))
        act, label = load_policy(a.run, None)
        frames, results = [], []
        for ep in range(a.episodes):
            obs, _ = env.reset(seed=a.seed + ep)
            done = False
            while not done:
                obs, _, term, trunc, info = env.step(act(obs))
                done = term or trunc
                f = env.render()
                frames.append(f)
                live.frame(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                live.status(drop=f"{ep + 1}/{a.episodes}", seconds=round(env.t * env.dt, 1), bounces=info["bounces"])
            results.append((env.t * env.dt, info["bounces"], info["cause"]))
            live.log(f"drop {ep + 1}: {env.t * env.dt:.1f} s, {info['bounces']} bounces, ended: {info['cause']}")
            frames.extend([frames[-1]] * 15)
        out = ROOT / "out" / "dashboard" / "videos" / f"{a.run}_seed{a.seed}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out, frames, fps=env.metadata["render_fps"], macro_block_size=1)
        live.done(video=str(out.relative_to(ROOT)), mean_seconds=float(np.mean([r[0] for r in results])))
    except Exception as e:
        live.failed(repr(e))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
