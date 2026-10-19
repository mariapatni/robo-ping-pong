"""Report a running job to the lab dashboard (lab/dashboard.py).

    from lab.live import Live
    live = Live("hover_touch", "Hover-the-ball touch test (sim)",
                "The paddle comes up flat under the ball and touches its bottom...", kind="calibration")
    live.status(phase="spot 3/20", sideways_mm=1.2)     # any numbers or short strings
    live.frame(bgr_image)                                # what the sim is showing right now
    live.point("loss", step, value)                      # a curve, drawn live on its card
    live.done(passed=True)                               # or live.failed("why")

Each run is a folder out/runs/<id>/ holding status.json, the latest frame.jpg
and curves.json; the dashboard reads the folder, so nothing is registered by
hand, and a job that dies without calling done() shows as stale once its
heartbeat stops. Frames are written at most `max_fps` times a second and
atomically (write, then rename), so the dashboard never reads half a JPEG.
Cheap enough to leave in every script: with no frames sent it is a small JSON
write per call.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "out" / "runs"
if "PYTEST_CURRENT_TEST" in os.environ:  # tests report to a throwaway folder, not the dashboard
    import tempfile
    RUNS = Path(tempfile.gettempdir()) / "lab_runs_test"


def _atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class Live:
    def __init__(self, name: str, title: str, description: str = "", kind: str = "sim",
                 run_id: str | None = None, max_fps: float = 10.0, command: str | None = None):
        self.id = run_id or f"{time.strftime('%Y%m%d_%H%M%S')}_{name}_{os.getpid()}"
        self.dir = RUNS / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._min_dt = 1.0 / max_fps
        self._last_frame = 0.0
        self._curves: dict[str, list] = {}
        self._last_curves = 0.0
        self.info = {
            "id": self.id, "name": name, "title": title, "description": description, "kind": kind,
            "command": command or " ".join(sys.argv), "pid": os.getpid(), "started": time.time(),
            "state": "running", "fields": {}, "log": [],
        }
        self._write()

    def _write(self) -> None:
        self.info["heartbeat"] = time.time()
        _atomic(self.dir / "status.json", json.dumps(self.info, default=float).encode())

    def status(self, **fields) -> None:
        self.info["fields"].update(fields)
        self._write()

    def log(self, line: str) -> None:
        """A line for the card's log (the last 200 are kept)."""
        self.info["log"] = (self.info["log"] + [line])[-200:]
        self._write()

    def due(self) -> bool:
        """Whether a frame sent now would be kept: render only when it is."""
        return time.time() - self._last_frame >= self._min_dt

    def frame(self, bgr, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_frame < self._min_dt:
            return
        import cv2
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            _atomic(self.dir / "frame.jpg", buf.tobytes())
            self._last_frame = now
            self.info["frames"] = self.info.get("frames", 0) + 1
            if now - self.info.get("heartbeat", 0) > 1.0:
                self._write()

    def point(self, curve: str, x: float, y: float) -> None:
        self._curves.setdefault(curve, []).append([float(x), float(y)])
        if time.time() - self._last_curves > 2.0:
            self.flush_curves()

    def flush_curves(self) -> None:
        _atomic(self.dir / "curves.json", json.dumps(self._curves).encode())
        self._last_curves = time.time()
        self._write()

    def done(self, **fields) -> None:
        self.info["fields"].update(fields)
        self.info["state"] = "done"
        self.info["ended"] = time.time()
        if self._curves:
            self.flush_curves()
        self._write()

    def failed(self, why: str) -> None:
        self.info["state"] = "failed"
        self.info["error"] = why
        self.info["ended"] = time.time()
        self._write()
