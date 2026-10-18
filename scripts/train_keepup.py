"""Train the keep-up policy: PPO, a small MLP, CPU only.

    .venv/bin/python scripts/train_keepup.py                 # 5M steps
    .venv/bin/python scripts/train_keepup.py --steps 1e6 --run smoke
    .venv/bin/python scripts/train_keepup.py --randomize --steps 1e7 --run v2

This is its own policy, kept apart from the return policy in
docs/reward-design.md: its own env (sim/keepup.py), its own reward, its own
checkpoints under policies/keepup/<run>/. Nothing here reads or writes the
return policy's weights or reward terms.

Measured before training, so the curve has something to be judged against:
holding the paddle level (a zero action) averages 3.9 bounces, and uniform
random actions 0.9. A policy that does not clear 3.9 has learned nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_env(rank: int, seed: int, randomize: bool = False, episode_s: float | None = None):
    def thunk():
        from sim.keepup import KeepUpConfig, KeepUpEnv

        over = {} if episode_s is None else {"episode_s": episode_s}
        env = KeepUpEnv(KeepUpConfig.randomized(**over) if randomize else KeepUpConfig(**over))
        env.reset(seed=seed + rank)
        return env

    return thunk


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=float, default=5e6)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run", default="default", help="subdirectory of policies/keepup/")
    p.add_argument("--randomize", action="store_true",
                   help="wide drops, randomised physics, sensor noise and pushes")
    p.add_argument("--resume", type=int, default=None,
                   help="continue this run from its checkpoint at this step count")
    p.add_argument("--start-stage", type=int, default=0, help="curriculum stage to start at")
    p.add_argument("--episode-s", type=float, default=None,
                   help="episode time limit; longer practises the drift that ends long rallies")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="discount; 0.99 looks ~3 s ahead at 30 Hz, 0.995 ~7 s")
    p.add_argument("--lr", type=float, default=3e-4, help="learning rate at the start of this run")
    p.add_argument("--lr-end", type=float, default=0.0,
                   help="learning rate at the end; tapering lets PPO settle instead of wobbling")
    p.add_argument("--eval-every", type=float, default=2e6, help="steps between keep-the-best evaluations")
    p.add_argument("--eval-episodes", type=int, default=48)
    p.add_argument("--eval-limit-s", type=float, default=300.0, help="time limit of an evaluation episode")
    p.add_argument("--max-stage", type=int, default=9,
                   help="last curriculum stage; past 5 almost nothing is catchable in time")
    args = p.parse_args(argv)

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

    torch.set_num_threads(1)  # the sim is the bottleneck; keep torch off its cores
    out = ROOT / "policies" / "keepup" / args.run
    out.mkdir(parents=True, exist_ok=True)

    venv = SubprocVecEnv([make_env(i, args.seed, args.randomize, args.episode_s) for i in range(args.envs)])
    venv = VecMonitor(venv, info_keywords=("bounces",))
    # Observations span joint angles in rad and ball velocities in m/s, so
    # they are normalised. Rewards are not: a bounce is worth 1 and that unit
    # is what makes the logged return readable.
    if args.resume:
        venv = VecNormalize.load(str(out / "ckpt" / f"keepup_vecnormalize_{args.resume}_steps.pkl"), venv)
        venv.training = True
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    class Bounces(BaseCallback):
        """Log what the task is actually about - bounces per episode, and why
        episodes end - rather than only the shaped return."""

        def __init__(self):
            super().__init__()
            self.causes = {}

        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.logger.record_mean("keepup/bounces", info["bounces"])
                    if "home_dist" in info:
                        self.logger.record_mean("keepup/home_dist_cm", 100 * info["home_dist"])
                        self.logger.record_mean("keepup/start_dist_cm", 100 * info["start_dist"])
                    if "apex_mean" in info:
                        self.logger.record_mean("keepup/apex_cm", 100 * info["apex_mean"])
                    cause = info.get("cause", "?")
                    self.causes[cause] = self.causes.get(cause, 0) + 1
            return True

        def _on_rollout_end(self) -> None:
            n = max(1, sum(self.causes.values()))
            for k, v in self.causes.items():
                self.logger.record(f"keepup/end_{k.replace(' ', '_')}", v / n)
            self.causes = {}

    class Curriculum(BaseCallback):
        """Widen where the ball drops from only once the policy has earned it.

        Stage 0 drops every ball at one spot, the centre. Each stage after
        adds a ring 6 cm wider, until the drops cover the whole footprint
        (0.54 m from the centre at its far corners, so 9 stages).

        A stage is judged on its NEWEST RING only: it is passed when the last
        `window` episodes whose drop came from that ring average `bar`
        bounces. The first version judged the mean over all episodes, and as
        only ~15% of drops came from the new ring, the easy distances carried
        the average - it passed stages 5 and 6 in 50k steps each without
        having learned them. The env also sends half the off-centre drops to
        the newest ring, so it is practised, not diluted.

        Gated on skill rather than on a clock: a timed
        schedule widens on time whether or not the policy can bounce yet, and
        a ball at the far edge of the horseshoe is almost never caught by a
        policy that cannot, so it teaches nothing.

        Writes the stage to curriculum.json in the run directory, so the live
        viewer can play the same drops training is on.
        """

        stages, ring, radius = 9, 0.06, 0.60
        bar, window = 15.0, 200  # stage 0: all episodes, all at the centre
        ring_bar, ring_window = 12.0, 100  # later stages: newest-ring episodes only

        def __init__(self, start_stage: int = 0, max_stage: int = 9):
            super().__init__()
            from collections import deque

            self.stages = max_stage
            self.stage, self.history = start_stage, []
            self.recent = deque(maxlen=self.window)

        def _on_training_start(self) -> None:
            self._apply()

        def _on_step(self) -> bool:
            inner = (self.stage - 1) * self.ring
            for info in self.locals["infos"]:
                if "episode" in info and (self.stage == 0 or info.get("spawn_dist", 0.0) > inner):
                    self.recent.append(info["bounces"])
            return True

        def _on_rollout_end(self) -> None:
            mean = float(np.mean(self.recent)) if self.recent else 0.0
            self.logger.record("keepup/stage", self.stage)
            self.logger.record("keepup/stage_mean_bounces", mean)
            need, bar = (self.window, self.bar) if self.stage == 0 else (self.ring_window, self.ring_bar)
            if self.stage < self.stages and len(self.recent) >= need and mean >= bar:
                self.stage += 1
                from collections import deque

                self.recent = deque(maxlen=self.ring_window)  # judged on its own ring
                self._apply()

        def _apply(self) -> None:
            scale = self.stage * self.ring / self.radius
            prev = max(0, self.stage - 1) * self.ring / self.radius
            self.training_env.set_attr("spawn_scale", scale)
            self.training_env.set_attr("spawn_scale_prev", prev)
            if self.stage > 0 and self.recent.maxlen != self.ring_window:
                from collections import deque  # a resumed run starting past stage 0

                self.recent = deque(maxlen=self.ring_window)
            self.history.append({"stage": self.stage, "spawn_scale": scale, "timesteps": self.num_timesteps})
            print(f"curriculum: stage {self.stage}/{self.stages} at {self.num_timesteps:,} steps", flush=True)
            (out / "curriculum.json").write_text(json.dumps(
                {"stage": self.stage, "stages": self.stages, "spawn_scale": scale,
                 "spawn_scale_prev": prev, "bar_bounces": self.bar if self.stage == 0 else self.ring_bar,
                 "history": self.history}, indent=2))

    class KeepBest(BaseCallback):
        """Score the policy every `--eval-every` steps and keep the best one.

        v7 showed why: with a fixed learning rate it plateaued and wobbled -
        checkpoints between 64 and 136 s median survival - and the final one
        landed on a dip. The last policy a run writes is not its best.

        Every evaluation plays the same `--eval-episodes` drops (fixed seeds),
        deterministically, on the full task with `--eval-limit-s` episodes, in
        its own environments so training's curriculum and statistics are not
        disturbed. The score is mean survival time. The best policy so far is
        saved as best.zip with its observation statistics, and every result is
        logged to best.json.
        """

        def __init__(self):
            super().__init__()
            self.last, self.best, self.history = -np.inf, -np.inf, []

        def _on_training_start(self) -> None:
            env = SubprocVecEnv([make_env(1000 + i, 0, args.randomize, args.eval_limit_s)
                                 for i in range(args.envs)])
            self.eval_env = VecNormalize(env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)

        def _on_step(self) -> bool:
            if self.num_timesteps - self.last >= args.eval_every:
                self.last = self.num_timesteps
                self._evaluate()
            return True

        def _on_training_end(self) -> None:
            self._evaluate()
            self.eval_env.close()

        def _evaluate(self) -> None:
            import copy

            self.eval_env.obs_rms = copy.deepcopy(venv.obs_rms)  # see what training sees
            self.eval_env.seed(12345)  # the same drops every time
            _, lengths = evaluate_policy(self.model, self.eval_env, n_eval_episodes=args.eval_episodes,
                                         deterministic=True, return_episode_rewards=True)
            secs = np.array(lengths) / 30.0  # ~30 decisions a second
            full = float(np.mean(secs >= 0.98 * args.eval_limit_s))
            rec = {"timesteps": int(self.num_timesteps), "mean_s": round(float(secs.mean()), 1),
                   "median_s": round(float(np.median(secs)), 1), "full": round(full, 3)}
            self.logger.record("keepup/eval_mean_s", rec["mean_s"])
            self.logger.record("keepup/eval_median_s", rec["median_s"])
            if rec["mean_s"] > self.best:
                self.best, rec["best"] = rec["mean_s"], True
                self.model.save(out / "best")
                venv.save(str(out / "best_vecnormalize.pkl"))
            self.history.append(rec)
            print(f"eval at {self.num_timesteps:,}: mean {rec['mean_s']} s, median {rec['median_s']} s, "
                  f"full {100 * full:.0f}%" + ("  <- best" if rec.get("best") else ""), flush=True)
            (out / "best.json").write_text(json.dumps({"best_mean_s": self.best, "history": self.history},
                                                      indent=2))

    def lr_schedule(start: int):
        """Linear from --lr to --lr-end over THIS run's steps.

        SB3 hands a schedule its progress over the whole step count, so a run
        resumed at 40M steps of 60M would start the taper a third of the way
        down. `p0` is that starting progress; this rescales from it.
        """
        p0 = args.steps / (start + args.steps)
        return lambda p: args.lr_end + (args.lr - args.lr_end) * float(np.clip(p / p0, 0.0, 1.0))

    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=1024,
        batch_size=2048,
        n_epochs=10,
        learning_rate=lr_schedule(0),
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        # Start near "hold level", which already earns ~4 bounces, rather than
        # thrashing at unit std and spending the first million steps
        # rediscovering stillness.
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), log_std_init=-1.0),
        seed=args.seed,
        verbose=1,
        tensorboard_log=None,
    )

    ckpt = CheckpointCallback(save_freq=max(1, 500_000 // args.envs), save_path=str(out / "ckpt"),
                              name_prefix="keepup", save_vecnormalize=True)
    if args.resume:
        model = PPO.load(out / "ckpt" / f"keepup_{args.resume}_steps", env=venv, device="cpu")
        model.gamma = model.rollout_buffer.gamma = args.gamma  # a resume may lengthen the horizon
        model.learning_rate = lr_schedule(model.num_timesteps)
        model._setup_lr_schedule()
        print(f"resumed {args.run} from {args.resume:,} steps", flush=True)
    callbacks = [Bounces(), ckpt, KeepBest()] + ([Curriculum(args.start_stage, args.max_stage)] if args.randomize else [])
    model.learn(total_timesteps=int(args.steps), callback=callbacks, progress_bar=False,
                reset_num_timesteps=not args.resume)

    model.save(out / "policy")
    venv.save(str(out / "vecnormalize.pkl"))
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"saved {out / 'policy.zip'}")
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
