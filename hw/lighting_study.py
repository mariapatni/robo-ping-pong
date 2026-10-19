"""How well the white-ball tracker holds up as the light changes.

    .venv/bin/python -m hw.lighting_study out/hw/clips/front_lit.mp4          # synthetic, from one clip
    .venv/bin/python -m hw.lighting_study --real out/hw/clips/*.mp4           # real clips side by side

Synthetic: the clip, recorded in good light, is tracked as it is - that track
is the reference - and then re-lit frame by frame in each of the ways below
and tracked again from scratch. Per condition: how many reference sightings
are still found (recall), how many sightings are somewhere else (false), the
centre and radius error, and how far the depth (which goes as 1/radius) moves.

    under_2x, under_4x   exposure too short: everything darker
    over_2x              too long: the ball and anything bright clip to white
    dim_noisy            a dim room the camera brightens with gain: noise x4
    warm                 tungsten bulbs, ~2700 K, white balance not following
    cool                 overcast daylight, ~7500 K
    overhead             a hard light above: bright at the top, dark below
    backlit              a bright window behind: auto-exposure meters for it,
                         the room goes dark and the window blooms
    haze                 low contrast: glare on the lens, or a washed-out screen

These are camera-side changes applied to the whole frame. They do not move
shadows or highlights on the ball itself, which real lights do - the --real
mode is for that: record the same throw under each real setup (blinds open
and shut, overhead on and off, a lamp behind the arm) with
`python -m hw.track --raw <name>`, and compare.

Writes out/hw/lighting/<stamp>/{report.txt, sheet.jpg}.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from .track import ClipReader
from .vision import ROOT, DetectorConfig, Tracker

CONDITIONS = ["baseline", "under_2x", "under_4x", "over_2x", "dim_noisy", "warm", "cool", "overhead",
              "backlit", "haze"]


def relight(img: np.ndarray, cond: str, rng: np.random.Generator) -> np.ndarray:
    # Work in roughly linear light, as the sensor sees it, then re-encode.
    lin = (img.astype(np.float32) / 255.0) ** 2.2
    h, w = img.shape[:2]
    if cond == "under_2x":
        lin *= 0.5
    elif cond == "under_4x":
        lin *= 0.25
    elif cond == "over_2x":
        lin *= 2.0
    elif cond == "dim_noisy":
        lin = lin * 0.25 + rng.normal(0, 0.004, lin.shape).astype(np.float32)  # read noise at low light
        lin *= 4.0  # the camera's gain brings it back up, noise and all
    elif cond == "warm":
        lin *= np.array([0.45, 0.75, 1.0], np.float32)  # BGR
    elif cond == "cool":
        lin *= np.array([1.0, 0.9, 0.7], np.float32)
    elif cond == "overhead":
        lin *= np.linspace(1.6, 0.5, h, dtype=np.float32)[:, None, None]
    elif cond == "backlit":
        lum = lin.mean(axis=2)
        bright = (lum > np.percentile(lum, 92)).astype(np.float32)
        glow = cv2.GaussianBlur(bright, (0, 0), 25)[..., None]
        lin = lin * 0.35 + 0.8 * glow + 0.6 * bright[..., None]
    elif cond == "haze":
        lin = lin * 0.55 + 0.25
    out = np.clip(lin, 0, 1) ** (1 / 2.2) * 255
    return out.astype(np.uint8)


BALL = "white"  # set from --ball


def run(frames: list[np.ndarray], times: list[float], cond: str, seed: int = 0) -> list[dict]:
    tracker = Tracker(cfg=DetectorConfig(ball=BALL))
    rng = np.random.default_rng(seed)
    out = []
    for f, t in zip(frames, times):
        r = tracker(relight(f, cond, rng) if cond != "baseline" else f, t)
        d = r["det"]
        out.append({"det": (d.u, d.v, d.r) if d is not None and r["used"] else None, "live": r["live"],
                    "bright": d.bright if d else np.nan, "moving": d.moving if d else np.nan})
    return out


def compare(ref: list[dict], got: list[dict]) -> dict:
    n_ref = found = false = 0
    centre, radius = [], []
    for a, b in zip(ref, got):
        if a["det"] is not None and a["live"]:
            n_ref += 1
            if b["det"] is not None:
                du = np.hypot(b["det"][0] - a["det"][0], b["det"][1] - a["det"][1])
                if du <= max(4.0, 0.6 * a["det"][2]):
                    found += 1
                    centre.append(du)
                    radius.append(b["det"][2] / a["det"][2] - 1)
                else:
                    false += 1
        elif b["det"] is not None and b["live"]:
            false += 1
    radius = np.array(radius)
    return {"ref": n_ref, "recall": found / max(n_ref, 1), "false": false,
            "centre_px": float(np.median(centre)) if centre else np.nan,
            "radius_err": float(np.median(np.abs(radius))) if len(radius) else np.nan,
            "depth_err": float(np.median(np.abs(1 / (1 + radius) - 1))) if len(radius) else np.nan,
            "bright_cue": float(np.nanmean([g["bright"] for g in got if g["det"]])) if any(g["det"] for g in got) else np.nan}


def load(path: Path, limit: int | None = None) -> tuple[list[np.ndarray], list[float]]:
    src = ClipReader(path)
    frames, times = [], []
    while limit is None or len(frames) < limit:
        try:
            f, t, _ = src.read()
        except EOFError:
            break
        frames.append(f)
        times.append(t)
    src.close()
    return frames, times


def sheet(frames, results: dict[str, list[dict]], k: int, rng) -> np.ndarray:
    tiles = []
    for cond, res in results.items():
        img = relight(frames[k], cond, rng) if cond != "baseline" else frames[k].copy()
        d = res[k]["det"]
        if d:
            cv2.circle(img, (int(d[0]), int(d[1])), int(d[2]) + 4, (0, 255, 0), 3)
        img = cv2.resize(img, (480, 270))
        cv2.putText(img, cond, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, cond, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clips", nargs="+", type=Path)
    p.add_argument("--real", action="store_true", help="compare real clips instead of re-lighting one")
    p.add_argument("--frames", type=int, default=None, help="use only the first N frames")
    p.add_argument("--ball", choices=("white", "orange"), default="white")
    args = p.parse_args(argv)
    global BALL
    BALL = args.ball
    out = ROOT / "out" / "hw" / "lighting" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    if args.real:
        lines.append(f"{'clip':24s} {'frames':>6s} {'found':>6s} {'tracked':>8s} {'radius':>7s} "
                     f"{'bright cue':>10s} {'motion cue':>10s}")
        tiles = {}
        for path in args.clips:
            frames, times = load(path, args.frames)
            res = run(frames, times, "baseline")
            found = np.mean([r["det"] is not None for r in res])
            live = np.mean([r["live"] for r in res])
            rad = np.nanmedian([r["det"][2] if r["det"] else np.nan for r in res]) if found else np.nan
            lines.append(f"{path.stem:24s} {len(res):6d} {found:6.0%} {live:8.0%} {rad:6.1f}px "
                         f"{np.nanmean([r['bright'] for r in res]):10.0%} {np.nanmean([r['moving'] for r in res]):10.0%}")
            k = next((i for i, r in enumerate(res) if r["det"]), len(res) // 2)
            tiles[path.stem] = (frames[k], res[k])
        imgs = []
        for name, (f, r) in tiles.items():
            img = f.copy()
            if r["det"]:
                cv2.circle(img, (int(r["det"][0]), int(r["det"][1])), int(r["det"][2]) + 4, (0, 255, 0), 3)
            img = cv2.resize(img, (480, 270))
            cv2.putText(img, name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            imgs.append(img)
        while len(imgs) % 3:
            imgs.append(np.zeros_like(imgs[0]))
        cv2.imwrite(str(out / "sheet.jpg"), np.vstack([np.hstack(imgs[i:i + 3]) for i in range(0, len(imgs), 3)]))
    else:
        frames, times = load(args.clips[0], args.frames)
        results = {c: run(frames, times, c) for c in CONDITIONS}
        ref = results["baseline"]
        n_ref = sum(r["det"] is not None and r["live"] for r in ref)
        lines.append(f"{args.clips[0].name}: {len(frames)} frames, ball tracked in {n_ref} of them (the reference)")
        lines.append(f"{'condition':10s} {'recall':>7s} {'false':>6s} {'centre':>8s} {'radius':>7s} {'depth':>7s} "
                     f"{'bright cue':>10s}")
        for cond, res in results.items():
            c = compare(ref, res)
            lines.append(f"{cond:10s} {c['recall']:7.0%} {c['false']:6d} {c['centre_px']:6.1f}px "
                         f"{c['radius_err']:7.1%} {c['depth_err']:7.1%} {c['bright_cue']:10.0%}")
        k = max(range(len(ref)), key=lambda i: (ref[i]["det"] is not None and ref[i]["live"], -abs(i - len(ref) // 2)))
        cv2.imwrite(str(out / "sheet.jpg"), sheet(frames, results, k, np.random.default_rng(0)))
    text = "\n".join(lines)
    print(text, flush=True)
    (out / "report.txt").write_text(text + "\n")
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
