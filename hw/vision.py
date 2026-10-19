"""The webcam's view of the ball: find it, place it in the arm's frame, filter it.

The keep-up policy needs six numbers from a camera - the ball's position
relative to the paddle face and its velocity - in the world frame of the sim
(origin at the arm's base, z up). Everything else it reads comes from the
servo encoders. This module turns frames into those six numbers:

    Camera        a background thread holding the newest frame and its time
    BallDetector  white, round, and (usually) moving: pixel centre and radius
    PaddleFinder  the blue blade, used to calibrate the camera with the arm
    CameraModel   pinhole camera placed in the world: pixels <-> metres
    BallFilter    Kalman filter under gravity: position and velocity

Depth comes from the ball's apparent size, since one camera cannot triangulate:
distance = f * R / r_px. That makes the axis pointing into the camera far
noisier than the two across the image, and BallFilter weights it accordingly.

Lighting: a white ball is the hard case. It is bright and colourless only when
lit from the front; against a bright window it is a grey disc darker than its
background. So a candidate is accepted on either of two cues - colourless and
brighter than its surroundings, or colourless-ish and moving - and then on
shape and size. See hw/lighting_study.py for how far each cue carries.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMERA_FILE = ROOT / "hw" / "camera.json"

# OpenCV numbers AVFoundation cameras differently from ffmpeg: on this Mac,
# OpenCV 1 is the FaceTime HD camera and 0 the iPhone (Continuity Camera).
CAMERA_INDEX = 1
WIDTH, HEIGHT, FPS = 1280, 720, 30

BALL_R = 0.020  # m: a regulation 40 mm ball
PADDLE_R = 0.055  # m: the 110 mm blade
GRAVITY = np.array([0.0, 0.0, -9.81])
REF_SPAN = 245.0  # a well-exposed frame's contrast range, Lab L 1st-99th percentile (this room: 248)
# Where the ball can be while it matters, in the world frame (m): within ~30 cm
# of the spot the keep-up policy bounces over (x 0.30, y 0), from the table up
# to well above the highest apex it aims for.
# A still candidate starts a track only with this much of a circle outlined
# around it (the real ball: 0.9; a square: well under half), and a size that
# matches a ball on the table at that pixel this closely.
STILL_OUTLINE = 0.6
STILL_SIZE_TOL = 0.2
WORKSPACE = (np.array([0.0, -0.30, -0.02]), np.array([0.60, 0.30, 0.85]))


# -- capture -------------------------------------------------------------------


class Camera:
    """Reads frames on a background thread; `read` returns the newest one.

    The policy runs at 30 Hz on whatever frame is newest, so frames are never
    queued: a slow consumer skips frames rather than falling behind.
    """

    def __init__(self, index: int = CAMERA_INDEX, width: int = WIDTH, height: int = HEIGHT, fps: int = FPS):
        self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"camera {index} gives no frames (in use, or no permission for this terminal?)")
        self._frame, self._t, self._n = frame, time.monotonic(), 1
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop:
            ok, frame = self.cap.read()
            t = time.monotonic()
            if ok:
                with self._lock:
                    self._frame, self._t, self._n = frame, t, self._n + 1

    def read(self, after: int = 0, timeout: float = 1.0) -> tuple[np.ndarray, float, int]:
        """(frame, capture time, frame number), waiting for one newer than `after`."""
        t0 = time.monotonic()
        while True:
            with self._lock:
                if self._n > after:
                    return self._frame, self._t, self._n
            if time.monotonic() - t0 > timeout:
                raise TimeoutError("camera stopped delivering frames")
            time.sleep(0.002)

    def close(self) -> None:
        self._stop = True
        self._thread.join(timeout=1.0)
        self.cap.release()


# -- the camera in the world -----------------------------------------------------


@dataclass
class CameraModel:
    """Pinhole camera, no distortion. R, t map world points into the camera
    frame (x right, y down, z forward): p_cam = R @ p_world + t."""

    f: float
    cx: float
    cy: float
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    width: int = WIDTH
    height: int = HEIGHT
    # The detector's radius times this is the ball's true image radius. Fitted
    # from balls at known places (hw.ball_point); 1 until then.
    radius_scale: float = 1.0

    @classmethod
    def guess(cls, width: int = WIDTH, height: int = HEIGHT, hfov_deg: float = 70.0) -> "CameraModel":
        """Uncalibrated: a typical laptop camera's field of view, placed at the origin."""
        return cls(f=0.5 * width / np.tan(np.radians(hfov_deg) / 2), cx=width / 2, cy=height / 2,
                   width=width, height=height)

    @property
    def calibrated(self) -> bool:
        return bool(np.any(self.t))

    @property
    def position(self) -> np.ndarray:
        """The camera's centre in the world."""
        return -self.R.T @ self.t

    def to_cam(self, p_world) -> np.ndarray:
        return np.asarray(p_world, float) @ self.R.T + self.t

    def project(self, p_world) -> tuple[np.ndarray, np.ndarray]:
        """Pixels (N, 2) and depth along the optical axis (N,) of world points (N, 3)."""
        pc = np.atleast_2d(self.to_cam(p_world))
        z = pc[:, 2]
        uv = np.c_[self.f * pc[:, 0] / z + self.cx, self.f * pc[:, 1] / z + self.cy]
        return uv, z

    def sphere_image(self, p_cam, ball_r: float = BALL_R) -> np.ndarray:
        """Where a ball centred at p_cam (camera frame) appears: (u, v, r) of
        the circle a detector fits to its outline.

        Off the optical axis a sphere images as an ellipse, stretched away from
        the centre of the image and shifted outward of its centre's projection:
        its outline is the cone of rays tangent to the sphere, cut by the image
        plane. 25 degrees off-axis the ellipse is ~5% bigger than on-axis, and
        5% on the radius is 5% on the depth - 25 mm at half a metre. So the
        outline is computed exactly: the conic q^T C q = 0 with
        C = w w^T - cos^2(a) I, w the direction to the centre and a the cone's
        half-angle; its centre and semi-axes; r is their mean, which is what a
        least-squares circle through an ellipse this round comes to."""
        p = np.asarray(p_cam, float)
        d = np.linalg.norm(p)
        w = p / d
        C = np.outer(w, w) - (1.0 - (ball_r / d) ** 2) * np.eye(3)
        A, b, c0 = C[:2, :2], C[:2, 2], C[2, 2]
        x0 = -np.linalg.solve(A, b)
        k = x0 @ A @ x0 - c0
        lam = np.linalg.eigvalsh(A)  # both negative for a ball in front of the camera
        axes = np.sqrt(k / lam)
        return np.array([self.f * x0[0] + self.cx, self.f * x0[1] + self.cy, self.f * axes.mean()])

    def ball_in_cam(self, u: float, v: float, r_px: float, ball_r: float = BALL_R) -> np.ndarray:
        """The ball's centre in the camera frame from its image centre and radius.

        A sphere of radius R at distance d subtends a half-angle asin(R/d); on
        the axis its image radius is f*tan of that, so d = R*sqrt(f^2+r^2)/r
        along the ray through the image centre. That is the start; a few
        Newton steps on `sphere_image` then take out the off-axis stretch."""
        r_px = r_px * self.radius_scale
        want = np.array([u, v, r_px])
        ray = np.array([(u - self.cx) / self.f, (v - self.cy) / self.f, 1.0])
        p = ray / np.linalg.norm(ray) * ball_r * np.hypot(self.f, r_px) / r_px
        for _ in range(6):
            err = self.sphere_image(p, ball_r) - want
            if np.abs(err).max() < 1e-7:
                break
            h = 1e-6 * np.linalg.norm(p)
            J = np.column_stack([(self.sphere_image(p + h * e, ball_r) - self.sphere_image(p, ball_r)) / h
                                 for e in np.eye(3)])
            p = p - np.linalg.solve(J, err)
        return p

    def ball_in_world(self, u: float, v: float, r_px: float, ball_r: float = BALL_R) -> np.ndarray:
        return self.R.T @ (self.ball_in_cam(u, v, r_px, ball_r) - self.t)

    def measurement_cov(self, p_world, px_sigma: float = 0.7, r_sigma: float = 0.5) -> np.ndarray:
        """Covariance (3x3, world) of a ball position measured from the image.

        Across the image an error of px_sigma pixels is Z*px_sigma/f metres.
        Along the ray depth goes as 1/r, so a radius error of r_sigma pixels
        is Z^2*r_sigma/(f*R) metres - at a metre, ~25x the lateral error."""
        z = float(self.to_cam(p_world)[2])
        lat = z * px_sigma / self.f
        dep = z * z * r_sigma / (self.f * BALL_R)
        return self.R.T @ np.diag([lat**2, lat**2, dep**2]) @ self.R

    def radius_px(self, p_world, ball_r: float = BALL_R) -> float:
        """The radius the detector should report for a ball at p_world."""
        return float(self.sphere_image(self.to_cam(p_world), ball_r)[2]) / self.radius_scale

    def roi(self, lo=None, hi=None, pad: int = 40) -> tuple[int, int, int, int] | None:
        """Pixel box (x0, y0, x1, y1) around the box in the world where the ball can
        matter; None when uncalibrated. Keeps people and windows out of the search."""
        if not self.calibrated:
            return None
        lo, hi = WORKSPACE if lo is None else (lo, hi)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        uv, z = self.project(corners)
        if np.all(z <= 0.05):
            return None
        if np.any(z <= 0.05):  # part of the box is beside or behind the camera
            return 0, 0, self.width, self.height
        x0, y0 = np.floor(uv.min(0)).astype(int) - pad
        x1, y1 = np.ceil(uv.max(0)).astype(int) + pad
        return max(0, x0), max(0, y0), min(self.width, x1), min(self.height, y1)

    # -- storage --

    def save(self, path: Path = CAMERA_FILE, **extra) -> None:
        d = asdict(self) | {"R": self.R.tolist(), "t": self.t.tolist()} | extra
        path.write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, path: Path = CAMERA_FILE) -> "CameraModel":
        if not path.exists():
            return cls.guess()
        d = json.loads(path.read_text())
        return cls(f=d["f"], cx=d["cx"], cy=d["cy"], R=np.array(d["R"]), t=np.array(d["t"]),
                   width=d["width"], height=d["height"], radius_scale=d.get("radius_scale", 1.0))


# -- finding the ball ------------------------------------------------------------


@dataclass
class DetectorConfig:
    # "white": colourless and lighter than its surroundings. "orange": a vivid
    # orange, which the room has none of - the wooden table and skin share its
    # hue but are half as vivid.
    ball: str = "white"
    orange_hue: tuple[int, int] = (3, 22)  # OpenCV hue (0-179): red-orange to orange-yellow
    orange_vivid: float = 0.38  # Lab chroma / L, white-balanced: ball ~0.5, wood and skin ~0.25
    orange_moving: float = 0.8  # a blurred ball mixes with what is behind it: this share of the bar
    orange_val: float = 0.30  # lightness, as a share of the frame's 99th percentile
    scale: float = 0.5  # search at half resolution, refine at full
    max_chroma: float = 32.0  # Lab a/b distance from grey (0-255 units): "white-ish".
    # Measured at the table: the ball under warm room light 21-30, the wood 35-44.
    min_contrast: float = 10.0  # L above the local background, for the brightness cue
    min_light: float = 0.55  # and at least this share of the frame's highlights (99th percentile L)
    background_px: int = 41  # local background window, in search pixels
    min_motion: float = 18.0  # change from the running background, for the motion cue
    motion_alpha: float = 0.08  # how fast the running background forgets
    r_min: float = 3.0  # ball radius bounds, full-resolution pixels
    r_max: float = 70.0
    min_circularity: float = 0.45
    max_elongation: float = 4.0  # motion blur stretches the ball along its path
    gate_px: float = 120.0  # search radius around a prediction, full-resolution pixels
    normalise_alpha: float = 0.2  # how fast the exposure and white-balance correction follows the frame


@dataclass
class Detection:
    u: float  # centre, full-resolution pixels
    v: float
    r: float  # radius across the blur: the ellipse's semi-minor axis
    score: float
    bright: float  # share of the blob passing the brightness cue
    moving: float  # share passing the motion cue
    elongation: float
    circularity: float = 1.0
    fill: float = 1.0  # blob area / its smallest enclosing circle's: 1 for a disc, 0.64 for a square
    outline: float = 0.0  # share of the fitted circle that lies on an edge in the image


class BallDetector:
    """Finds the ball in one frame. Keeps a running background for the motion
    cue, so feed it consecutive frames."""

    def __init__(self, cfg: DetectorConfig | None = None):
        self.cfg = cfg or DetectorConfig()
        self.bg: np.ndarray | None = None
        self.debug: dict[str, np.ndarray] = {}
        self.candidates: list[Detection] = []
        self._norm: np.ndarray | None = None
        self._gains: np.ndarray | None = None

    def reset(self) -> None:
        self.bg = None
        self._norm = None
        self._gains = None

    def __call__(self, frame: np.ndarray, roi=None, predict=None, r_expect: float | None = None,
                 mask_out: np.ndarray | None = None, min_moving: float = 0.0,
                 max_candidates: int = 12) -> Detection | None:
        """`roi` (x0, y0, x1, y1) limits the search; `predict` (u, v) and
        `r_expect` (pixels) favour candidates where the filter expects the ball.
        `mask_out` (full-resolution bool) hides pixels, e.g. the paddle.
        `min_moving` rejects blobs with less than that share of moving pixels.
        Returns the best candidate; all of them, best first, are left in
        `self.candidates` for a caller that can rule some out."""
        c = self.cfg
        s = c.scale
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(small, (3, 3), 0)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, a, b = lab[..., 0], lab[..., 1] - 128.0, lab[..., 2] - 128.0
        span, white = self._exposure(L, a, b)
        # Colourless relative to the scene's own white, so a warm or cool light
        # does not push the ball over the threshold; and contrast and motion in
        # proportion to the frame's contrast, so exposure and haze do not either.
        chroma = np.hypot(a - white[0], b - white[1])
        k = c.background_px | 1
        contrast = (L - cv2.blur(L, (k, k))) * (REF_SPAN / span)

        f32 = blur.astype(np.float32)
        if self.bg is None or self.bg.shape != f32.shape:
            self.bg = f32.copy()
        motion = np.abs(f32 - self.bg).max(axis=2) * (REF_SPAN / span)
        cv2.accumulateWeighted(f32, self.bg, c.motion_alpha)

        if c.ball == "orange":
            # The static cue is the colour itself; "bright" keeps its name below.
            # Vividness is Lab chroma over lightness, in white-balanced colour:
            # it holds still as the exposure changes, and it separates the ball
            # (~0.5) from the wood and from skin (both ~0.25), which share its
            # hue. HSV saturation does not: skin reaches 190, and in shadow
            # noise it reads 255.
            bal = self._balance(blur)
            H = cv2.cvtColor(bal, cv2.COLOR_BGR2HSV)[..., 0]
            lab_b = cv2.cvtColor(bal, cv2.COLOR_BGR2LAB).astype(np.float32)
            Lb = lab_b[..., 0]
            vivid = np.hypot(lab_b[..., 1] - 128.0, lab_b[..., 2] - 128.0) / np.maximum(Lb, 1.0)
            hue_ok = (H >= c.orange_hue[0]) & (H <= c.orange_hue[1])
            lit = Lb >= c.orange_val * np.percentile(Lb, 99)
            # Above the orange-ish things already in the scene: 1.3x the 95th
            # percentile of their vividness, within [0.12, orange_vivid]. Haze
            # washes the table out with the ball, and the bar comes down with it.
            same_hue = vivid[hue_ok & lit]
            bar = c.orange_vivid if same_hue.size < 50 else float(np.clip(1.3 * np.percentile(same_hue, 95), 0.12,
                                                                           c.orange_vivid))
            bright = hue_ok & lit & (vivid >= bar)
            moving = hue_ok & lit & (vivid >= c.orange_moving * bar) & (motion > c.min_motion)
        else:
            colourless = chroma < c.max_chroma
            # Light in absolute terms as well as against its surroundings, scaled to
            # the frame's own highlights so it follows the exposure.
            light = L > c.min_light * np.percentile(L, 99)
            bright = colourless & light & (contrast > c.min_contrast)
            moving = (chroma < 1.6 * c.max_chroma) & (motion > c.min_motion)

        allowed = np.ones_like(bright, bool)
        if roi is not None:
            x0, y0, x1, y1 = (int(round(v * s)) for v in roi)
            allowed[:] = False
            allowed[y0:y1, x0:x1] = True
        if predict is not None:
            yy, xx = np.ogrid[:bright.shape[0], :bright.shape[1]]
            gate = (xx - predict[0] * s) ** 2 + (yy - predict[1] * s) ** 2 < (c.gate_px * s) ** 2
            allowed &= gate
        if mask_out is not None:
            allowed &= ~cv2.resize(mask_out.astype(np.uint8), (bright.shape[1], bright.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
        self.debug = {"bright": bright, "moving": moving}

        # Each cue is searched on its own. Joined, a white wall that passes the
        # brightness test swallows a ball that only the motion test separates.
        found = []
        for cue in (moving, bright):
            cand = (cue & allowed).astype(np.uint8)
            # Cut one-pixel threads (a table edge's highlight) that join the ball
            # to its surroundings; anything three pixels across survives.
            cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
            cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            found += self._blobs(cand, bright, moving, predict, r_expect, min_moving)
        # The same ball found by both cues: keep the better-scored one.
        found.sort(key=lambda c: -c[0])
        kept = []
        for f in found:
            if all(np.hypot(f[1] - k[1], f[2] - k[2]) > max(f[3], k[3]) for k in kept):
                kept.append(f)
        self.candidates = []
        for score, u, v, r, fb, fm, elong, circ, fill in kept[:max_candidates]:
            # A streak has no round outline to fit: its centre is the ball at
            # mid-exposure and its width the ball's diameter.
            support = 0.0
            if not (fm > 0.5 and elong > 1.4):
                u, v, r, support = self._refine(frame, u, v, r)
            self.candidates.append(Detection(u, v, r, float(score), fb, fm, float(elong), float(circ), float(fill),
                                             support))
        return self.candidates[0] if self.candidates else None

    def _balance(self, img: np.ndarray, update: bool = True) -> np.ndarray:
        """White-balanced: scaled per channel so the frame's lightest unclipped
        surfaces come out grey (smoothed over frames). Under warm bulbs
        everything turns orange, the wooden table most of all; this undoes it."""
        f = img.astype(np.float32)
        grey = f.mean(axis=2)
        light = (grey >= np.percentile(grey, 95)) & (f.max(axis=2) < 250)
        if update and light.sum() > 50:
            white = f[light].mean(axis=0)
            gains = white.mean() / np.maximum(white, 1.0)
            k = self.cfg.normalise_alpha
            self._gains = gains if self._gains is None else (1 - k) * self._gains + k * gains
        if self._gains is None:
            return img
        return np.clip(f * self._gains, 0, 255).astype(np.uint8)

    def _ball_image(self, win: np.ndarray) -> np.ndarray:
        """A grey image in which the ball is the bright thing: plain brightness
        for a white ball; for an orange one, saturation where the hue is orange."""
        if self.cfg.ball != "orange":
            return cv2.cvtColor(win, cv2.COLOR_BGR2GRAY)
        bal = self._balance(win, update=False)
        H = cv2.cvtColor(bal, cv2.COLOR_BGR2HSV)[..., 0]
        lab = cv2.cvtColor(bal, cv2.COLOR_BGR2LAB).astype(np.float32)
        vivid = np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0) / np.maximum(lab[..., 0], 1.0)
        hue_ok = (H >= self.cfg.orange_hue[0]) & (H <= self.cfg.orange_hue[1])
        return np.where(hue_ok, np.clip(vivid * 400.0, 0, 255), 0).astype(np.uint8)

    def _exposure(self, L, a, b) -> tuple[float, np.ndarray]:
        """The frame's contrast range (Lab L, 1st to 99th percentile) and the
        colour of its white (mean a, b of its lightest unclipped surfaces),
        smoothed over frames."""
        lo, hi = np.percentile(L, (1, 99))
        light = (L >= np.percentile(L, 95)) & (L < 250)
        white = np.array([a[light].mean(), b[light].mean()]) if light.sum() > 50 else np.zeros(2)
        # A white point far from grey is a coloured wall, not a tinted light.
        white *= min(1.0, 25.0 / max(np.hypot(*white), 1e-6))
        params = np.r_[max(hi - lo, 25.0), white]
        k = self.cfg.normalise_alpha
        self._norm = params if self._norm is None else (1 - k) * self._norm + k * params
        return float(self._norm[0]), self._norm[1:]

    def _blobs(self, cand, bright, moving, predict, r_expect, min_moving) -> list[tuple]:
        c, s = self.cfg, self.cfg.scale
        out = []
        n, labels, stats, cents = cv2.connectedComponentsWithStats(cand, connectivity=8)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            r_eq = np.sqrt(area / np.pi) / s
            if not (c.r_min * 0.7 <= r_eq <= c.r_max * 1.5):
                continue
            x, y, w, h = stats[i, :4]
            blob = labels[y:y + h, x:x + w] == i
            cnts, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt = max(cnts, key=cv2.contourArea)
            perim = cv2.arcLength(cnt, True)
            circ = 4 * np.pi * area / max(perim * perim, 1e-6)
            fill = area / max(np.pi * cv2.minEnclosingCircle(cnt)[1] ** 2, 1e-6)
            if len(cnt) >= 5:
                (ex, ey), (d1, d2), _ = cv2.fitEllipse(cnt)
                minor, major = sorted((d1, d2))
                r = 0.5 * minor / s
                elong = major / max(minor, 1e-6)
            else:
                r, elong = r_eq, 1.0
            if not (c.r_min <= r <= c.r_max) or elong > c.max_elongation:
                continue
            fb = float(bright[y:y + h, x:x + w][blob].mean())
            fm = float(moving[y:y + h, x:x + w][blob].mean())
            if fm < min_moving:
                continue
            # A blurred ball is an elongated blob, so judge its roundness on
            # what blur leaves: the ellipse fit, not the raw circularity.
            if circ < c.min_circularity and not (fm > 0.5 and elong > 1.3 and circ > 0.25):
                continue
            score = 1.5 * min(circ, 1.0) + fb + fm - 0.15 * (elong - 1.0)
            u, v = cents[i] / s
            if predict is not None:
                d = np.hypot(u - predict[0], v - predict[1]) / c.gate_px
                score -= 1.5 * d * d
            if r_expect:
                score -= 2.0 * abs(np.log(r / r_expect))
            out.append((score, u, v, r, fb, fm, elong, circ, fill))
        return out

    def _refine(self, frame, u, v, r):
        """Centre and radius at full resolution, from the ball's outline.

        A circle is fitted to the window's edges (Hough), then refined to
        sub-pixel by least squares on the edge points near it - a pixel of
        radius is 5% of the distance. The outline is used rather than a
        threshold because a lit ball's top and sides stay sharp while its
        shaded underside fades into a warm table. A blurred ball has no
        circular outline; it falls back to thresholding."""
        pad = int(max(8, 2.2 * r))
        x0, y0 = max(0, int(u) - pad), max(0, int(v) - pad)
        x1, y1 = min(frame.shape[1], int(u) + pad), min(frame.shape[0], int(v) + pad)
        win = frame[y0:y1, x0:x1]
        if win.shape[0] < 8 or win.shape[1] < 8:
            return u, v, r
        g = cv2.GaussianBlur(self._ball_image(win), (5, 5), 1.2)
        if r >= 6:
            circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=1e4, param1=60, param2=12,
                                       minRadius=int(0.6 * r), maxRadius=int(np.ceil(1.4 * r)))
            if circles is not None:
                cu, cv_, cr = circles[0][0]
                if np.hypot(cu + x0 - u, cv_ + y0 - v) < 0.8 * r:
                    return self._fit_edges(g, cu, cv_, cr, x0, y0)
        return (*self._refine_threshold(frame, u, v, r), 0.0)

    @staticmethod
    def _fit_edges(g, cu, cv_, cr, x0, y0):
        edges = cv2.Canny(g, 30, 90)
        ys, xs = np.nonzero(edges)
        d = np.hypot(xs - cu, ys - cv_)
        near = np.abs(d - cr) < max(2.0, 0.12 * cr)
        if near.sum() < 12:
            return cu + x0, cv_ + y0, float(cr), 0.0
        # Algebraic circle fit: x^2 + y^2 + D x + E y + F = 0.
        X, Y = xs[near].astype(float), ys[near].astype(float)
        A = np.c_[X, Y, np.ones_like(X)]
        D, E, F = np.linalg.lstsq(A, -(X * X + Y * Y), rcond=None)[0]
        fu, fv = -D / 2, -E / 2
        fr = np.sqrt(max(fu * fu + fv * fv - F, 0.0))
        if not (0.85 * cr < fr < 1.15 * cr):
            fu, fv, fr = cu, cv_, float(cr)
        # How much of the circle is outlined: the share of 48 directions with
        # an edge within two pixels of it. A ball shows most of its outline; a
        # square touches its circle in four places.
        dist = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
        ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        px = np.clip((fu + fr * np.cos(ang)).astype(int), 0, g.shape[1] - 1)
        py = np.clip((fv + fr * np.sin(ang)).astype(int), 0, g.shape[0] - 1)
        support = float((dist[py, px] <= 2.0).mean())
        return fu + x0, fv + y0, float(fr), support

    def _refine_threshold(self, frame, u, v, r):
        """Threshold a window around the blob between the ball and its
        surround, on whiteness, and fit an ellipse; its minor axis is the
        radius across any motion blur."""
        pad = int(max(6, 2.2 * r))
        x0, y0 = max(0, int(u) - pad), max(0, int(v) - pad)
        x1, y1 = min(frame.shape[1], int(u) + pad), min(frame.shape[0], int(v) + pad)
        win = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2LAB)
        if win.size == 0:
            return u, v, r
        win = win.astype(np.float32)
        L = win[..., 0] - 2.0 * np.hypot(win[..., 1] - 128.0, win[..., 2] - 128.0)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = (xx - u) ** 2 + (yy - v) ** 2 < (0.6 * r) ** 2
        ring = ((xx - u) ** 2 + (yy - v) ** 2 > (1.5 * r) ** 2)
        if inside.sum() < 3 or ring.sum() < 3:
            return u, v, r
        lin, lout = np.median(L[inside]), np.median(L[ring])
        if abs(lin - lout) < 6:  # no edge to fit; keep the coarse estimate
            return u, v, r
        m = ((L - lout) * np.sign(lin - lout) > 0.5 * abs(lin - lout)).astype(np.uint8)
        n, labels = cv2.connectedComponents(m)
        lab = labels[int(np.clip(v - y0, 0, m.shape[0] - 1)), int(np.clip(u - x0, 0, m.shape[1] - 1))]
        if lab == 0:
            return u, v, r
        cnts, _ = cv2.findContours((labels == lab).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        if len(cnt) < 5:
            return u, v, r
        (ex, ey), (d1, d2), _ = cv2.fitEllipse(cnt)
        r2 = 0.5 * min(d1, d2)
        if not (0.6 * r < r2 < 1.6 * r):
            return u, v, r
        return ex + x0, ey + y0, r2


class PaddleFinder:
    """The blue blade: the largest saturated-blue blob."""

    def __init__(self, hue=(102, 125), min_sat: int = 90, min_val: int = 40):
        self.hue, self.min_sat, self.min_val = hue, min_sat, min_val

    def mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, (self.hue[0], self.min_sat, self.min_val), (self.hue[1], 255, 255))
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0

    def __call__(self, frame: np.ndarray, min_area: int = 300):
        """(u, v, area, major_px) of the blade, or None."""
        m = self.mask(frame).astype(np.uint8)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(m)
        if n < 2:
            return None
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            return None
        cnts, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        major = max(cv2.fitEllipse(cnt)[1]) if len(cnt) >= 5 else 2 * np.sqrt(area / np.pi)
        return float(cents[i][0]), float(cents[i][1]), area, float(major)


# -- filtering -------------------------------------------------------------------------


class BallFilter:
    """Constant-velocity Kalman filter with gravity, in the world frame.

    Bounces and hits change the velocity in one frame. When a measurement falls
    outside the gate, it is tried again with the velocity's uncertainty opened
    up to a hit's size; if it fits then, it is a bounce and is accepted.
    """

    def __init__(self, accel_sigma: float = 4.0, hit_sigma: float = 3.0, hit_side_sigma: float = 0.8,
                 gate: float = 16.0, max_coast: float = 0.25, confirm: int = 3):
        self.accel_sigma, self.hit_sigma, self.gate, self.max_coast = accel_sigma, hit_sigma, gate, max_coast
        self.hit_side_sigma = hit_side_sigma
        self.confirm, self.hits = confirm, 0
        self.floor = BALL_R  # the ball's centre resting on the table (z = 0)
        self.x: np.ndarray | None = None
        self.P = np.eye(6)
        self.t = 0.0
        self.t_seen = -np.inf
        self.bounces = 0

    @property
    def tracking(self) -> bool:
        """Following something, confirmed or not: predictions are worth making."""
        return self.x is not None and self.t - self.t_seen <= self.max_coast

    @property
    def live(self) -> bool:
        """Following something that has moved like a ball for `confirm` frames."""
        return self.tracking and self.hits >= self.confirm

    @property
    def pos(self) -> np.ndarray:
        return self.x[:3]

    @property
    def vel(self) -> np.ndarray:
        return self.x[3:]

    def predict(self, t: float) -> None:
        if self.x is None:
            return
        dt = max(t - self.t, 0.0)
        self.t = t
        if dt == 0:
            return
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        self.x = F @ self.x + np.r_[0.5 * GRAVITY * dt * dt, GRAVITY * dt]
        if self.x[2] < self.floor:  # the table holds it up
            self.x[2] = self.floor
            self.x[5] = max(self.x[5], 0.0)
        q = self.accel_sigma**2
        Q = np.zeros((6, 6))
        Q[:3, :3] = q * dt**4 / 4 * np.eye(3)
        Q[:3, 3:] = Q[3:, :3] = q * dt**3 / 2 * np.eye(3)
        Q[3:, 3:] = q * dt**2 * np.eye(3)
        self.P = F @ self.P @ F.T + Q

    def update(self, t: float, z: np.ndarray, Rm: np.ndarray, Rm_wide: np.ndarray | None = None) -> bool:
        """Fold in one measured position; False if it did not fit (left out).

        `Rm_wide` is the same measurement with its weak axis (depth) trusted
        less still. A measurement that misses the gate is tried, in order:
        as a depth outlier (Rm_wide), as a hit (the velocity allowed to jump,
        mostly vertically), and as both."""
        if self.x is None or t - self.t_seen > self.max_coast:
            self.x = np.r_[z, 0.0, 0.0, 0.0]
            self.P = np.diag(np.r_[np.diag(Rm), [2.0**2] * 3])
            self.t = self.t_seen = t
            self.hits = 1
            return True
        dt = max(t - self.t, 0.0)
        self.predict(t)
        # A hit during the frame: an unknown velocity change - a bounce
        # reverses the vertical, with only a little sideways - which has also
        # moved the ball by up to dt times that change by now.
        dv = np.diag([self.hit_side_sigma**2, self.hit_side_sigma**2, self.hit_sigma**2])
        hit = np.block([[dt * dt * dv, dt * dv], [dt * dv, dv]])
        wide = Rm if Rm_wide is None else Rm_wide
        for P, R, is_hit in ((self.P, Rm, False), (self.P, wide, False), (self.P + hit, Rm, True),
                             (self.P + hit, wide, True)):
            S = P[:3, :3] + R
            y = z - self.x[:3]
            if y @ np.linalg.solve(S, y) <= self.gate:
                self.bounces += is_hit
                K = P[:, :3] @ np.linalg.inv(S)
                self.x = self.x + K @ y
                self.P = (np.eye(6) - K @ np.eye(3, 6)) @ P
                self.t_seen = t
                self.hits += 1
                return True
        return False

    def predicted_px(self, cam: CameraModel, t: float) -> tuple[tuple[float, float], float] | None:
        """Where the ball should appear at time t, and how big; None if not tracking."""
        if not self.tracking:
            return None
        dt = t - self.t
        p = self.x[:3] + self.x[3:] * dt + 0.5 * GRAVITY * dt * dt
        p[2] = max(p[2], self.floor)
        uv, z = cam.project(p)
        if z[0] <= 0.05:
            return None
        return (float(uv[0, 0]), float(uv[0, 1])), cam.radius_px(p)


# -- the policy's ball inputs --------------------------------------------------------


class Tracker:
    """Frame in, ball state out: detector + camera model + filter, one call per frame."""

    def __init__(self, cam: CameraModel | None = None, cfg: DetectorConfig | None = None,
                 paddle: PaddleFinder | None = None):
        self.cam = cam or CameraModel.load()
        self.detector = BallDetector(cfg)
        self.filter = BallFilter()
        self.paddle = paddle or PaddleFinder()
        self.roi = self.cam.roi()
        # A track starts only on something moving unless the camera is placed:
        # then the 3D check rules out the static white things instead, and a
        # ball resting on the table can be picked up.
        self.acquire_moving = 0.0 if self.cam.calibrated else 0.3

    def resting_on_table(self, c: Detection, strict: bool = True) -> np.ndarray | None:
        """A still candidate starts a track only as a ball sitting on the table.
        Then its distance is known twice over - where its line of sight meets
        the table, and from its size - and the two must agree. A sheet of paper,
        a mug or a mark on the arm fails one or the other. Returns the ball's
        centre from the line of sight (the better of the two), or None."""
        cam = self.cam
        if strict and c.outline < STILL_OUTLINE:
            return None
        ray = cam.R.T @ np.array([(c.u - cam.cx) / cam.f, (c.v - cam.cy) / cam.f, 1.0])
        o = cam.position
        if ray[2] >= -1e-6:
            return None  # looking up: the table is not along this ray
        p = o + (BALL_R - o[2]) / ray[2] * ray
        if abs(c.r / cam.radius_px(p) - 1) > STILL_SIZE_TOL * (1 if strict else 1.5):
            return None
        return p

    def __call__(self, frame: np.ndarray, t: float) -> dict:
        pred = self.filter.predicted_px(self.cam, t)
        # The blade is blue, so it can never pass the colourless test - but a
        # moving blade's edge can, and the ball sitting on it touches it. Hide it.
        hide = self.paddle.mask(frame)
        if pred is not None:
            det = self.detector(frame, roi=self.roi, predict=pred[0], r_expect=pred[1], mask_out=hide)
        else:
            det = self.detector(frame, roi=self.roi, mask_out=hide, min_moving=self.acquire_moving)
        z, used, resting = None, False, False
        if det is not None and self.cam.calibrated:
            # The first candidate whose size puts it somewhere the ball can be:
            # this rules out people, lamps and marks on the arm. Starting a
            # track on something still also takes a properly round outline -
            # with no motion blur to excuse it, a square of paper is not a ball.
            det = None
            # Already following a ball on the table: keep placing it by its line
            # of sight, so it does not flicker between the two ways of placing it.
            on_table = self.filter.tracking and self.filter.pos[2] < BALL_R + 0.02
            for c in self.detector.candidates:
                # Still and sitting on the table: place it by its line of sight.
                p = self.resting_on_table(c, strict=not on_table) if c.moving < 0.3 else None
                resting = p is not None
                if p is None:
                    if pred is None and c.moving < 0.3:
                        continue  # a still thing that is not a ball on the table
                    p = self.cam.ball_in_world(c.u, c.v, c.r)
                if p is not None and np.all((WORKSPACE[0] <= p) & (p <= WORKSPACE[1])):
                    det, z = c, p
                    break
        elif det is not None:
            z = self.cam.ball_in_world(det.u, det.v, det.r)
        if det is not None:
            # On the table, the line of sight fixes it to a few millimetres; off
            # it, depth from size is the weak axis.
            if resting:
                cov = np.diag([0.004**2, 0.004**2, 0.002**2])
            else:
                # How well the radius is known. A sharp, still ball's outline
                # fits to a fraction of a pixel; a moving one is a smear whose
                # width wanders by 15% or more. Told so, the filter leans on the
                # ball's arc under gravity for depth, which fixes it in a few frames.
                r_sigma = max(0.5, (0.25 if det.moving > 0.3 else 0.04) * det.r)
                cov = self.cam.measurement_cov(z, r_sigma=r_sigma)
                wide = self.cam.measurement_cov(z, r_sigma=3 * r_sigma)
            used = self.filter.update(t, z, cov, None if resting else wide)
        else:
            self.filter.predict(t)
        live = self.filter.live
        return {"t": t, "det": det, "meas": z, "used": used, "live": live,
                "pos": self.filter.pos.copy() if live else None,
                "vel": self.filter.vel.copy() if live else None, "pred": pred}
