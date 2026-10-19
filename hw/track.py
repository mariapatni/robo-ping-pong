"""Track the ball live from the webcam, and record clips to test the tracker on.

    .venv/bin/python -m hw.track                      # track for 30 s, write an annotated video
    .venv/bin/python -m hw.track --seconds 20 --raw overhead_on   # also keep the raw frames
    .venv/bin/python -m hw.track --clip out/hw/clips/overhead_on.mp4   # re-run on a recorded clip
    .venv/bin/python -m hw.track --arm               # also read the arm: ball relative to the paddle

Writes out/hw/track_<stamp>/: track.mp4 (frames with the detection, the
filter's prediction and the 3D state drawn on), track.npz (per frame: time,
pixel detection, measured and filtered position, velocity) and summary.txt.
A still is kept at out/hw/track_latest.jpg while it runs.

--raw <name> saves the unannotated frames to out/hw/clips/<name>.mp4 with their
timestamps alongside, for hw.lighting_study.

With --arm the joint angles are read each frame and the paddle face placed by
the sim's kinematics, giving the six numbers the keep-up policy reads: ball
position relative to the face, and ball velocity.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .vision import ROOT, Camera, CameraModel, DetectorConfig, Tracker

CLIPS = ROOT / "out" / "hw" / "clips"


def draw(frame: np.ndarray, r: dict, roi, trail: list, rel=None) -> np.ndarray:
    img = frame.copy()
    if roi:
        cv2.rectangle(img, roi[:2], roi[2:], (90, 90, 90), 1)
    for a, b in zip(trail[:-1], trail[1:]):
        cv2.line(img, a, b, (0, 200, 255), 2)
    if r["pred"]:
        (pu, pv), pr = r["pred"]
        cv2.circle(img, (int(pu), int(pv)), max(int(pr), 2), (255, 160, 0), 1)
    d = r["det"]
    if d:
        colour = (0, 255, 0) if r["used"] else (0, 0, 255)
        cv2.circle(img, (int(d.u), int(d.v)), max(int(d.r), 2), colour, 2)
        cv2.putText(img, f"r {d.r:.1f}px  bright {d.bright:.0%}  moving {d.moving:.0%}",
                    (int(d.u) + 12, int(d.v) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    lines = []
    if r["live"]:
        p, v = r["pos"], r["vel"]
        lines.append(f"ball  x {p[0]:+.3f}  y {p[1]:+.3f}  z {p[2]:+.3f} m")
        lines.append(f"vel   x {v[0]:+.2f}  y {v[1]:+.2f}  z {v[2]:+.2f} m/s")
        if rel is not None:
            lines.append(f"ball - face  {rel[0]:+.3f} {rel[1]:+.3f} {rel[2]:+.3f} m")
    else:
        lines.append("no ball")
    panel = img[:12 + 26 * len(lines), :470]
    panel[:] = (0.35 * panel).astype(np.uint8)
    for k, s in enumerate(lines):
        cv2.putText(img, s, (12, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return img


class ClipReader:
    """A recorded clip played back as if it were the camera."""

    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        ts = path.with_suffix(".json")
        self.times = json.loads(ts.read_text())["t"] if ts.exists() else None
        self.k = 0

    def read(self, after: int = 0, timeout: float = 1.0):
        ok, frame = self.cap.read()
        if not ok:
            raise EOFError
        t = self.times[self.k] if self.times and self.k < len(self.times) else self.k / 30.0
        self.k += 1
        return frame, t, self.k

    def close(self):
        self.cap.release()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--clip", type=Path, help="run on a recorded clip instead of the camera")
    p.add_argument("--raw", help="also save unannotated frames as out/hw/clips/<name>.mp4")
    p.add_argument("--arm", action="store_true", help="read the arm and report the ball relative to the face")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--ball", choices=("white", "orange"), default="white")
    args = p.parse_args(argv)

    out = ROOT / "out" / "hw" / f"track_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    src = ClipReader(args.clip) if args.clip else Camera()
    tracker = Tracker(cfg=DetectorConfig(ball=args.ball))
    cam = tracker.cam
    print("camera", "calibrated" if cam.calibrated else "NOT calibrated (positions are in the camera's frame, "
          "with a guessed focal length)", flush=True)

    arm = world = None
    if args.arm:
        import mujoco

        from sim.paddle import face_pose

        from .so101 import SO101
        arm = SO101()
        world = arm.check.world

    raw = video = None
    raw_times: list[float] = []
    log = {k: [] for k in ("t", "det", "meas", "pos", "vel", "used", "q", "face")}
    trail: list[tuple[int, int]] = []
    n, t_start, latest = 0, None, 0.0
    try:
        while True:
            try:
                frame, t, n = src.read(after=n)
            except EOFError:
                break
            t_start = t if t_start is None else t_start
            if t - t_start > args.seconds and not args.clip:
                break
            r = tracker(frame, t)
            rel = face = q = None
            if arm is not None:
                q = arm.q()
                world.reset(q)
                mujoco.mj_forward(world.model, world.data)
                face = face_pose(world.model, world.data)[0]
                rel = r["pos"] - face if r["live"] else None
            d = r["det"]
            if d and r["used"]:
                trail = (trail + [(int(d.u), int(d.v))])[-20:]
            elif not r["live"]:
                trail = []
            log["t"].append(t - t_start)
            log["det"].append([d.u, d.v, d.r, d.score, d.bright, d.moving] if d else [np.nan] * 6)
            log["meas"].append(r["meas"] if r["meas"] is not None else [np.nan] * 3)
            log["pos"].append(r["pos"] if r["live"] else [np.nan] * 3)
            log["vel"].append(r["vel"] if r["live"] else [np.nan] * 3)
            log["used"].append(r["used"])
            log["q"].append(q if q is not None else [np.nan] * 5)
            log["face"].append(face if face is not None else [np.nan] * 3)

            if args.raw:
                if raw is None:
                    CLIPS.mkdir(parents=True, exist_ok=True)
                    raw = cv2.VideoWriter(str(CLIPS / f"{args.raw}.mp4"), cv2.VideoWriter_fourcc(*"avc1"),
                                          30, (frame.shape[1], frame.shape[0]))
                raw.write(frame)
                raw_times.append(t - t_start)
            if not args.no_video:
                img = draw(frame, r, tracker.roi, trail, rel)
                if video is None:
                    video = cv2.VideoWriter(str(out / "track.mp4"), cv2.VideoWriter_fourcc(*"avc1"),
                                            30, (img.shape[1] // 2, img.shape[0] // 2))
                video.write(cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
                if time.monotonic() - latest > 0.5:
                    cv2.imwrite(str(ROOT / "out" / "hw" / "track_latest.jpg"), img)
                    latest = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        for w in (raw, video):
            if w is not None:
                w.release()
        if args.raw:
            (CLIPS / f"{args.raw}.json").write_text(json.dumps({"t": raw_times}))

    arr = {k: np.array(v, float) for k, v in log.items()}
    np.savez(out / "track.npz", **arr)
    text = summary(arr)
    print(text, flush=True)
    (out / "summary.txt").write_text(text + "\n")
    print("wrote", out, flush=True)
    return 0


def summary(a: dict) -> str:
    t = a["t"]
    if len(t) < 2:
        return "no frames"
    dt = np.diff(t)
    found = ~np.isnan(a["det"][:, 0])
    live = ~np.isnan(a["pos"][:, 0])
    return "\n".join([
        f"{len(t)} frames in {t[-1]:.1f} s: {1 / np.median(dt):.1f} fps median, "
        f"longest gap {1000 * dt.max():.0f} ms",
        f"ball found in {found.mean():.0%} of frames, used by the filter in {a['used'].mean():.0%}, "
        f"tracked in {live.mean():.0%}",
    ] + ([f"radius {np.nanmedian(a['det'][:, 2]):.1f} px median "
          f"(brightness cue {np.nanmean(a['det'][:, 4]):.0%}, motion cue {np.nanmean(a['det'][:, 5]):.0%})"]
         if found.any() else []))


if __name__ == "__main__":
    raise SystemExit(main())
