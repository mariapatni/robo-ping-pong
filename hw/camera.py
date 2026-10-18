"""One camera, behind one interface, so the camera can be swapped by name.

    from hw.camera import open_camera
    with open_camera("facetime") as cam:
        frame, t = cam.wait_new()          # BGR image, capture time (time.monotonic())

Everything about a particular camera lives in hw/cameras/<name>.json - which
device, which mode, its measured quirks and, once hw/calibrate_camera.py has
run, its lens calibration. Code downstream (the ball detector, the 3D filter,
the live control loop, the simulated camera used for training) reads only
that file, so swapping cameras is: plug it in, add a JSON, calibrate it.

Backends:
  avfoundation  a macOS camera, by name or index (FaceTime HD, an iPhone over
                Continuity, a USB/UVC camera)
  file          a recorded video, for developing the tracker offline

A background thread reads frames as they arrive and keeps only the newest, so
a slow consumer never works on a stale queued frame. Each frame is stamped
with its arrival time; the camera's own latency (capture to arrival) is
measured separately and stored in the JSON as `latency_s`.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CAMERAS = Path(__file__).resolve().parent / "cameras"


@dataclass
class CameraSpec:
    name: str
    backend: str  # "avfoundation" or "file"
    device: str | int  # avfoundation: device name or index; file: path
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    latency_s: float | None = None  # capture to arrival, once measured
    calibration: dict | None = None  # written by hw/calibrate_camera.py
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, name_or_path: str | Path) -> "CameraSpec":
        p = Path(name_or_path)
        if not p.suffix:
            p = CAMERAS / f"{name_or_path}.json"
        d = json.loads(p.read_text())
        known = {k: d.pop(k) for k in list(d) if k in cls.__dataclass_fields__}
        return cls(**known, extra=d)

    def save(self, path: Path | None = None) -> Path:
        path = path or CAMERAS / f"{self.name}.json"
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "extra"} | self.extra
        path.write_text(json.dumps(d, indent=2) + "\n")
        return path

    def intrinsics(self) -> tuple[np.ndarray, np.ndarray]:
        """(K, distortion) from the calibration. Raises if not yet calibrated."""
        if not self.calibration:
            raise ValueError(f"camera {self.name!r} is not calibrated: run python -m hw.calibrate_camera")
        return np.array(self.calibration["K"], float), np.array(self.calibration["dist"], float)


def avfoundation_devices() -> dict[str, int]:
    """Video devices macOS lists, name -> index (ffmpeg's numbering, which
    OpenCV's AVFoundation backend shares). Indices change when devices come and
    go - an iPhone in range adds one - which is why specs name the device."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                       capture_output=True, text=True)
    devices, video = {}, False
    for line in r.stderr.splitlines():
        if "video devices" in line:
            video = True
        elif "audio devices" in line:
            video = False
        elif video and (m := re.search(r"\[(\d+)\] (.+)$", line)):
            devices[m.group(2).strip()] = int(m.group(1))
    return devices


class Camera:
    """The newest frame from a camera, with its arrival time."""

    def __init__(self, spec: CameraSpec):
        import cv2

        self.spec = spec
        if spec.backend == "avfoundation":
            index = spec.device if isinstance(spec.device, int) else avfoundation_devices().get(spec.device)
            if index is None:
                raise RuntimeError(f"no camera named {spec.device!r}; macOS lists {list(avfoundation_devices())}")
            self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, spec.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, spec.height)
            self.cap.set(cv2.CAP_PROP_FPS, spec.fps)
            self.live = True
        elif spec.backend == "file":
            self.cap = cv2.VideoCapture(str(spec.device))
            self.live = False
        else:
            raise ValueError(f"unknown camera backend {spec.backend!r}")
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera {spec.name!r} ({spec.backend}: {spec.device})")
        self._lock = threading.Condition()
        self._frame, self._t, self._n = None, 0.0, 0
        self._stop = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        t0 = time.monotonic()
        period = 1.0 / (self.cap.get(5) or self.spec.fps)  # 5 = CAP_PROP_FPS
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                if self.live:
                    continue
                break  # end of file
            # A file has no arrival time; give it the time it would have had.
            t = time.monotonic() if self.live else t0 + self._n * period
            with self._lock:
                self._frame, self._t, self._n = frame, t, self._n + 1
                self._lock.notify_all()
        with self._lock:
            self._stop = True
            self._lock.notify_all()

    def read(self) -> tuple[np.ndarray | None, float]:
        """The newest frame and its time, without waiting."""
        with self._lock:
            return self._frame, self._t

    def wait_new(self, timeout: float = 2.0) -> tuple[np.ndarray | None, float]:
        """Block until a frame newer than the last one returned, or None at the end."""
        with self._lock:
            n = getattr(self, "_seen", 0)
            self._lock.wait_for(lambda: self._n > n or self._stop, timeout=timeout)
            self._seen = self._n
            return (self._frame, self._t) if self._n > n else (None, self._t)

    @property
    def frames_read(self) -> int:
        return self._n

    def close(self) -> None:
        self._stop = True
        self._thread.join(timeout=2.0)
        self.cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_camera(name_or_spec: str | Path | CameraSpec) -> Camera:
    spec = name_or_spec if isinstance(name_or_spec, CameraSpec) else CameraSpec.load(name_or_spec)
    return Camera(spec)


def measure_rate(name: str, seconds: float = 3.0) -> float:
    """Frames per second actually delivered - often below the mode's nominal
    rate, because auto-exposure lengthens exposures in dim light."""
    with open_camera(name) as cam:
        cam.wait_new()
        n0, t0 = cam.frames_read, time.monotonic()
        time.sleep(seconds)
        return (cam.frames_read - n0) / (time.monotonic() - t0)
