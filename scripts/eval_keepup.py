"""Evaluate the keep-up policy, and render it.

    .venv/bin/python scripts/eval_keepup.py --run v1                 # 50 episodes
    .venv/bin/python scripts/eval_keepup.py --run v1 --video out/keepup.mp4
    .venv/bin/python scripts/eval_keepup.py --baseline zero --video out/keepup_hold.mp4
    .venv/bin/python scripts/eval_keepup.py --run v1 --randomize   # how brittle is it?

Reports bounces per episode and why each episode ended, the same way
scripts/evaluate.py breaks down the return task: a mean says whether it got
better, the causes say what to fix next. `--baseline zero` runs "hold the
paddle level" through the identical harness, which is the bar to beat.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.keepup import KeepUpConfig, KeepUpEnv  # noqa: E402


def load_policy(run: str, ckpt: str | None):
    """(act, label): a function obs -> action, using the run's own obs stats."""
    import pickle

    from stable_baselines3 import PPO

    d = ROOT / "policies" / "keepup" / run
    if ckpt == "best":  # the best policy a run's keep-the-best evaluations found
        weights, stats = d / "best", d / "best_vecnormalize.pkl"
    elif ckpt:  # a mid-training checkpoint, by step count
        weights, stats = d / "ckpt" / f"keepup_{ckpt}_steps", d / "ckpt" / f"keepup_vecnormalize_{ckpt}_steps.pkl"
    else:
        weights, stats = d / "policy", d / "vecnormalize.pkl"
    model = PPO.load(weights, device="cpu")
    with open(stats, "rb") as f:
        norm = pickle.load(f)

    def act(obs):
        o = np.clip((obs - norm.obs_rms.mean) / np.sqrt(norm.obs_rms.var + norm.epsilon),
                    -norm.clip_obs, norm.clip_obs)
        return model.predict(o.astype(np.float32), deterministic=True)[0]

    return act, f"policy {run}" + (f" @ {ckpt} steps" if ckpt else "")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="v1")
    p.add_argument("--ckpt", default=None, help="checkpoint step count, e.g. 1000000, or 'best'")
    p.add_argument("--baseline", choices=["zero", "random"], default=None)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=10_000, help="held out from training seeds")
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--video-episodes", type=int, default=3)
    p.add_argument("--size", type=int, nargs=2, default=(960, 720))
    p.add_argument("--randomize", action="store_true", help="evaluate in the randomised world")
    p.add_argument("--json", type=Path, default=None, help="also write every episode's result here")
    p.add_argument("--episode-s", type=float, default=None,
                   help="time limit per episode; long limits measure how long it really lasts")
    args = p.parse_args(argv)

    over = {} if args.episode_s is None else {"episode_s": args.episode_s}
    cfg = KeepUpConfig.randomized(**over) if args.randomize else KeepUpConfig(**over)
    env = KeepUpEnv(cfg,
                    render_size=tuple(args.size) if args.video else None)
    if args.baseline == "zero":
        act, label = (lambda o: np.zeros(5)), "hold level (zero action)"
    elif args.baseline == "random":
        act, label = (lambda o: env.action_space.sample()), "random actions"
    else:
        act, label = load_policy(args.run, args.ckpt)

    bounces, causes, frames, apex, angle, home, start, spawn, ends, secs = [], Counter(), [], [], [], [], [], [], [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        record = args.video is not None and ep < args.video_episodes
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(act(obs))
            done = term or trunc
            if record:
                frames.append(env.render())
        if record and frames:
            frames.extend([frames[-1]] * 15)  # a half-second hold between episodes
        bounces.append(info["bounces"])
        secs.append(env.t * env.dt)
        causes[info["cause"]] += 1
        spawn.append(info.get("spawn_dist", 0.0))
        ends.append(info["cause"])
        if "apex_mean" in info:
            apex.append(info["apex_mean"])
            angle.append(info["angle_mean"])
        if "home_dist" in info:
            home.append(info["home_dist"])
            start.append(info["start_dist"])

    b = np.array(bounces)
    world = "randomised world" if args.randomize else "fixed world"
    print(f"{label}: {args.episodes} episodes, {world}")
    print(f"  bounces  mean {b.mean():.1f}  median {np.median(b):.0f}  min {b.min()}  max {b.max()}")
    T = np.array(secs)
    limit = env.cfg.episode_s
    print(f"  lasted   median {np.median(T):.1f} s, mean {T.mean():.1f} s  (limit {limit:.0f} s)")
    for mark in (20, 60, 120, 300):
        if mark <= limit:
            print(f"           {mark:>4.0f} s or more: {np.mean(T >= mark - 1e-6):4.0%}")
    if apex:
        print(f"  apex     mean {100 * np.mean(apex):.1f} cm above the face  (target band 20-35)")
        print(f"  takeoff  mean {np.mean(angle):.1f} deg off vertical")
    if home:
        print(f"  centre   first catch {100 * np.mean(start):.1f} cm from it, "
              f"last 5 bounces {100 * np.mean(home):.1f} cm from it")
    if max(spawn) > 0.03:
        sp, bo = np.array(spawn), np.array(bounces)
        print("  by drop distance from the centre:")
        for lo, hi in ((0.0, 0.03), (0.03, 0.12), (0.12, 0.20), (0.20, 0.40)):
            m = (sp >= lo) & (sp < hi)
            if m.any():
                full = np.mean([causes_i == "timeout" for causes_i in np.array(ends)[m]])
                print(f"    {100 * lo:4.0f}-{100 * hi:<3.0f} cm  n={m.sum():3d}  bounces mean {bo[m].mean():5.1f}"
                      f"  median {np.median(bo[m]):4.0f}  lasted to the limit {100 * full:3.0f}%")
    for cause, n in causes.most_common():
        print(f"  {cause:<12} {n:>3}  ({100 * n / args.episodes:.0f}%)")

    if args.json:
        import json

        args.json.write_text(json.dumps([{"seconds": t, "bounces": int(b), "cause": c, "spawn_dist": d}
                                         for t, b, c, d in zip(secs, bounces, ends, spawn)]))
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.video, frames, fps=env.metadata["render_fps"], macro_block_size=1)
        print(f"wrote {args.video}  ({len(frames)} frames)")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
