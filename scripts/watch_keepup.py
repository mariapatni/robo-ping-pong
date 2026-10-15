"""Watch the keep-up policy live, while it trains.

    .venv/bin/mjpython scripts/watch_keepup.py --run v2 --randomize

Training runs headless in subprocesses, so it cannot be watched directly. This
plays the newest checkpoint of a run in the interactive viewer, in real time,
and swaps in each newer checkpoint as training writes it - so the window shows
the policy improving as it learns. Needs `mjpython`, not `python` (see README).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_keepup import load_policy  # noqa: E402
from sim.keepup import KeepUpConfig, KeepUpEnv  # noqa: E402


def newest(run: str) -> str | None:
    """'final' if the run has finished, else the latest checkpoint's step count."""
    d = ROOT / "policies" / "keepup" / run
    if (d / "policy.zip").exists():
        return "final"
    steps = [int(m.group(1)) for p in (d / "ckpt").glob("keepup_*_steps.zip")
             if (m := re.match(r"keepup_(\d+)_steps", p.stem))]
    return str(max(steps)) if steps else None


def curriculum(run: str) -> dict | None:
    """The training curriculum's current stage, if the run writes one."""
    import json

    p = ROOT / "policies" / "keepup" / run / "curriculum.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="v2")
    p.add_argument("--randomize", action="store_true", help="play in the randomised world")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed, 1.0 = real time")
    args = p.parse_args(argv)

    env = KeepUpEnv(KeepUpConfig.randomized() if args.randomize else None)
    loaded, act, ep = None, None, 0
    with mujoco.viewer.launch_passive(env.world.model, env.world.data) as viewer:
        cam = env.camera()
        viewer.cam.lookat[:] = cam.lookat
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = cam.distance, cam.azimuth, cam.elevation
        while viewer.is_running():
            ck = newest(args.run)
            if ck is None:
                # Nothing trained yet. Play the scene anyway, arm held still, so
                # the window shows the drops instead of sitting blank.
                act, loaded = (lambda o: np.zeros(5)), None
            elif ck != loaded:
                act, _ = load_policy(args.run, None if ck == "final" else ck)
                loaded = ck
                print(f"loaded {args.run} @ {ck if ck == 'final' else f'{int(ck):,} steps'}", flush=True)

            # Drop from where training currently is, so each widening it earns
            # shows up here too. A finished run plays the full footprint.
            stage = curriculum(args.run)
            if stage and ck != "final":
                env.spawn_scale = stage["spawn_scale"]
            else:
                env.spawn_scale = 1.0
            obs, _ = env.reset(seed=50_000 + ep)
            ep += 1
            done = False
            while not done and viewer.is_running():
                tick = time.perf_counter()
                obs, _, term, trunc, info = env.step(act(obs))
                done = term or trunc
                label = ("waiting for first checkpoint - arm held still" if loaded is None
                         else loaded if loaded == "final" else f"{int(loaded) / 1e6:.1f}M steps")
                if stage and ck != "final":
                    label += f"\ncurriculum stage {stage['stage']}/{stage['stages']}"
                try:
                    viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                      f"{args.run} @ {label}\nepisode {ep}",
                                      f"bounces {info['bounces']}"))
                except AttributeError:
                    pass  # older viewer without text overlays
                viewer.sync()
                time.sleep(max(0.0, env.dt / args.speed - (time.perf_counter() - tick)))
            print(f"episode {ep}: {info['bounces']} bounces, ended by {info.get('cause', 'viewer closed')}", flush=True)
            time.sleep(0.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
