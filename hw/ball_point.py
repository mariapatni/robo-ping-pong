"""Hold the ball up; the arm puts the paddle under it. Then show it where the
ball really was, and the camera learns from the difference.

    .venv/bin/python -m hw.ball_point                 # until Ctrl-C, then refit the camera
    .venv/bin/python -m hw.ball_point --points 8      # stop after 8 corrections
    .venv/bin/python -m hw.ball_point --refit         # refit from the last session's points only

It talks (macOS `say`), so it can be run from across the room. Each round:

  1. Hold the ball still over the table, somewhere the paddle can reach.
  2. The camera places it in the arm's world, and the arm moves the centre of
     its paddle, face up, HOVER metres directly under it.
  3. Rest the ball on the centre of the paddle and hold it there lightly. Now
     its true position is known from the arm (face centre + one radius along
     the face normal), and what the camera sees of it is recorded beside that.
  4. Lift it away and hold it somewhere new.

Step 2 is the test - how far off was the paddle? - and step 3 the ground
truth. On Ctrl-C the arm folds away, and the camera's placement, focal length
and a radius scale (how the detector's ball size compares to the true one) are
refitted to the corrections together with the blade sightings from
hw.cam_calibrate. hw/camera.json is rewritten unless --no-write; the old one
is kept in out/hw/ball_point/.

Writes out/hw/ball_point/<stamp>/{points.json, NN_*.jpg, report.txt}.
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import subprocess
import time

import cv2
import numpy as np

from .vision import (BALL_R, CAMERA_FILE, ROOT, WORKSPACE, BallDetector, Camera, CameraModel, DetectorConfig,
                     PaddleFinder)

OUT = ROOT / "out" / "hw" / "ball_point"
HOVER = 0.05  # m between the ball's underside and the paddle face when the arm arrives
FACE_Z = (0.10, 0.32)  # the face heights the arm is sent to
REACH = (0.15, 0.44)  # m from the base: where the arm holds its face level (sim.keepup.face_up_footprint)
STILL_S = 1.0  # how long the ball must stay still to count as held
STILL_PX = 2.5  # ... moving no more than this (std of its centre, pixels)


_voice: subprocess.Popen | None = None


def say(text: str, wait: bool = False) -> None:
    """Speak, after whatever is still being said; with `wait`, until it is said."""
    global _voice

    def finish(proc):
        # macOS `say` now and then never returns; a hung voice must not stall
        # the session behind it. Nothing it says takes longer than this.
        try:
            proc.wait(timeout=2.0 + 0.1 * len(text))
        except subprocess.TimeoutExpired:
            proc.kill()

    if _voice is not None:
        finish(_voice)
    _voice = subprocess.Popen(["say", "-r", "190", text])
    if wait:
        finish(_voice)
    try:
        print(">>", text, flush=True)
    except (BrokenPipeError, OSError):
        pass


class Watcher:
    """Follows one ball by its last sighting, and says when it has been held still.

    Simpler than vision.Tracker on purpose: a ball held in a hand neither falls
    nor bounces, so the gravity filter has nothing to offer here."""

    def __init__(self, cam: CameraModel, ball: str = "white"):
        self.cam = cam
        self.det = BallDetector(DetectorConfig(ball=ball))
        self.paddle = PaddleFinder()
        self.hist: collections.deque = collections.deque()
        self.last = None  # (u, v, r) of the last sighting
        self.t_last = -np.inf

    def __call__(self, frame, t, near=None) -> object:
        """One frame. `near` (u, v, r) points the search at where the ball should be."""
        hide = self.paddle.mask(frame)
        guide = near or self.last
        if guide is not None:
            d = self.det(frame, predict=guide[:2], r_expect=guide[2], mask_out=hide)
        else:
            d = self.det(frame, mask_out=hide, min_moving=0.3)  # pick it up as it moves into place
        if d is not None and near is None:
            p = self.cam.ball_in_world(d.u, d.v, d.r)
            lo, hi = WORKSPACE
            if not np.all((lo - 0.1 <= p) & (p <= hi + 0.1)):
                d = None
        if d is None:
            if t - self.t_last > 0.5:
                self.last = None
                self.hist.clear()
            return None
        self.last, self.t_last = (d.u, d.v, d.r), t
        self.hist.append((t, d.u, d.v, d.r))
        while self.hist and t - self.hist[0][0] > STILL_S:
            self.hist.popleft()
        return d

    def still(self) -> tuple[float, float, float] | None:
        """(u, v, r) averaged over the last second, if the ball has been held still for it."""
        h = np.array(self.hist)
        if len(h) < 0.7 * 30 * STILL_S or h[-1, 0] - h[0, 0] < 0.8 * STILL_S:
            return None
        if h[:, 1].std() > STILL_PX or h[:, 2].std() > STILL_PX or h[:, 3].std() > 0.08 * h[:, 3].mean():
            return None
        return float(h[:, 1].mean()), float(h[:, 2].mean()), float(np.median(h[:, 3]))


def face(world, q):
    import mujoco

    from sim.paddle import face_pose
    world.reset(np.asarray(q, float))
    mujoco.mj_forward(world.model, world.data)
    return face_pose(world.model, world.data)


def plan_under(arm, ball: np.ndarray):
    """Joint angles putting the face centre, level, HOVER under the ball; or a reason it cannot."""
    import mujoco

    from sim.paddle import FACE_SITE
    from sim.pose import solve_face_pose_ik
    world, guard = arm.check.world, arm.check.guard
    reach = float(np.hypot(ball[0], ball[1]))
    if ball[0] < 0.05 or not (REACH[0] <= reach <= REACH[1]):
        return None, None, f"that is {reach * 100:.0f} cm from the base; hold it {REACH[0] * 100:.0f} to " \
                           f"{REACH[1] * 100:.0f} cm out, in front of the arm"
    if ball[2] < BALL_R + HOVER + FACE_Z[0] - 0.02:
        return None, None, "that is too low; hold it higher"
    target = np.array([ball[0], ball[1], np.clip(ball[2] - BALL_R - HOVER, *FACE_Z)])
    sid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_SITE, FACE_SITE)
    s = solve_face_pose_ik(world, target, (0.0, 0.0, 1.0), site=sid, guard=guard, q0=arm.q(), seed=0,
                           restarts=3, iters=250)
    if s.position_error > 0.01 or s.normal_error_deg > 6:
        return None, target, f"out of reach ({s.position_error * 100:.0f} cm short)"
    if not guard.is_safe(s.qpos) or not arm.check.path_ok(arm.q(), s.qpos):
        return None, target, "the way there runs into the table"
    return s.qpos.copy(), target, ""


def snapshot(path, frame, cam: CameraModel, marks: list[tuple[np.ndarray, tuple, str]]) -> None:
    img = frame.copy()
    for p, colour, label in marks:
        (uv,), z = cam.project(np.asarray(p)[None])
        if z[0] > 0:
            cv2.drawMarker(img, (int(uv[0]), int(uv[1])), colour, cv2.MARKER_CROSS, 22, 2)
            cv2.putText(img, label, (int(uv[0]) + 12, int(uv[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2,
                        cv2.LINE_AA)
    cv2.imwrite(str(path), img)


PHASES = {  # banner title, what to do, colour (BGR)
    "start": ("STARTING", "the arm is unfolding", (120, 120, 120)),
    "hold": ("HOLD THE BALL STILL", "anywhere over the table the paddle can reach", (0, 200, 255)),
    "moving": ("MOVING UNDER THE BALL", "keep holding it where it is", (255, 160, 0)),
    "correct": ("REST THE BALL ON THE PADDLE", "on the centre of the blue face; hold it lightly", (0, 220, 0)),
    "returning": ("RETURNING", "lift the ball away", (180, 180, 180)),
    "count": ("PICK A NEW SPOT", "", (255, 0, 255)),
    "done": ("DONE", "folding the arm", (120, 120, 120)),
}


def draw_state(frame: np.ndarray, cam: CameraModel, S: dict) -> np.ndarray:
    """The live view: phase, the ball as the camera sees it, the paddle, history."""
    img = frame.copy()
    h, w = img.shape[:2]
    title, hint, colour = PHASES[S["phase"]]
    if S["phase"] == "count":
        hint = f"hold it still in {max(0, int(np.ceil(S['count_end'] - time.monotonic())))} s"
    if S["phase"] == "hold" and S.get("why"):
        hint = S["why"]
    cv2.rectangle(img, (0, 0), (w, 64), (20, 20, 20), -1)
    cv2.rectangle(img, (0, 0), (14, 64), colour, -1)
    cv2.putText(img, title, (28, 32), cv2.FONT_HERSHEY_DUPLEX, 0.95, colour, 2, cv2.LINE_AA)
    cv2.putText(img, hint, (28, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)

    def mark(p, col, label, kind=cv2.MARKER_CROSS):
        if p is None:
            return
        (uv,), z = cam.project(np.asarray(p, float)[None])
        if z[0] > 0 and 0 <= uv[0] < w and 0 <= uv[1] < h:
            cv2.drawMarker(img, (int(uv[0]), int(uv[1])), col, kind, 24, 2)
            cv2.putText(img, label, (int(uv[0]) + 14, int(uv[1]) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1,
                        cv2.LINE_AA)

    mark(S.get("face"), (255, 160, 0), "paddle centre", cv2.MARKER_TILTED_CROSS)
    if S["phase"] in ("moving", "correct"):
        mark(S.get("ball_est"), (0, 200, 255), "ball, as seen when held")
    if S["phase"] == "correct":
        mark(S.get("truth"), (0, 220, 0), "rest it here", cv2.MARKER_DIAMOND)

    d = S.get("det")
    if d is not None and time.monotonic() - S.get("t_det", 0) < 0.3:
        u, v, r = d
        cv2.circle(img, (int(u), int(v)), int(r) + 3, (0, 255, 0), 2)
        prog = S.get("still", 0.0)
        cv2.rectangle(img, (int(u - r), int(v + r + 8)), (int(u + r), int(v + r + 14)), (60, 60, 60), -1)
        cv2.rectangle(img, (int(u - r), int(v + r + 8)), (int(u - r + 2 * r * prog), int(v + r + 14)),
                      (0, 255, 0), -1)
    xyz = S.get("xyz")
    lines = [f"ball (camera)  x {xyz[0]:+.3f}  y {xyz[1]:+.3f}  z {xyz[2]:+.3f} m" if xyz is not None and
             time.monotonic() - S.get("t_det", 0) < 0.3 else "ball (camera)  not seen",
             f"rounds {S['rounds']}" + (f"   last error {S['last_err']}" if S.get("last_err") else ""),
             f"camera f {cam.f:.0f} px, radius scale {cam.radius_scale:.2f}"]
    y0 = h - 16 - 24 * (len(lines) - 1)
    cv2.rectangle(img, (0, y0 - 22), (560, h), (20, 20, 20), -1)
    for k, line in enumerate(lines):
        cv2.putText(img, line, (12, y0 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    return img


def session(args) -> list[dict]:
    from sim.keepup import KeepUpEnv

    from .so101 import SO101
    from .viewer import Painter, Viewer
    out = OUT / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    cam = CameraModel.load()
    camera = Camera()
    S = {"phase": "start", "rounds": 0}
    viewer = Viewer(args.port)
    painter = Painter(camera, viewer, lambda f: draw_state(f, cam, S))
    print("live view at", viewer.url, flush=True)
    if not args.no_browser:
        subprocess.Popen(["open", viewer.url])
    arm = SO101()
    world = arm.check.world
    w = Watcher(cam, args.ball)
    pts: list[dict] = []
    n = 0
    ready = KeepUpEnv().q0

    def seen(d, t):
        """Record a sighting for the view, and how long it has been held still."""
        if d is not None:
            S["det"], S["t_det"] = (d.u, d.v, d.r), time.monotonic()
            S["xyz"] = cam.ball_in_world(d.u, d.v, d.r)
        hist = len(w.hist) / (30 * STILL_S)
        S["still"] = min(1.0, hist) if w.still() is not None or hist < 1 else 0.6

    def set_face():
        S["face"] = face(world, arm.q())[0]

    arm.configure(acceleration=30, torque_limit=500)
    arm.torque(True)
    try:
        arm.unfold()
        arm.move_to(ready, speed=0.4)
        set_face()
        S["phase"] = "hold"
        say("Ready. Hold the ball still over the table.")
        plan_q = plan_t = ball_est = None
        t_state = time.monotonic()
        last_said = None
        while args.points is None or len(pts) < args.points:
            frame, t, n = camera.read(after=n)
            phase = S["phase"]
            if phase == "hold":
                seen(w(frame, t), t)
                held = w.still()
                if held is None:
                    continue
                ball_est = cam.ball_in_world(*held)
                plan_q, plan_t, why = plan_under(arm, ball_est)
                if plan_q is None:
                    S["why"] = f"can't get under that: {why}. Try closer."
                    say(f"I can't get under that, {why}. Try somewhere closer.", wait=True)
                    S["phase"] = "returning"
                    continue
                S["why"] = ""
                S["ball_est"], S["phase"] = ball_est, "moving"
                say("Got it. Moving.")
                print(f"ball seen at {np.round(ball_est, 3)}; face to {np.round(plan_t, 3)}", flush=True)
                try:
                    arm.move_to(plan_q, speed=0.5)
                except RuntimeError as e:  # refused by the sim check, or stalled
                    print("move failed:", e, flush=True)
                    say("I couldn't get there safely. Let's try another spot.", wait=True)
                    S["phase"] = "returning"
                    continue
                set_face()
                frame, t, n = camera.read(after=n)
                snapshot(out / f"{len(pts):02d}_under.jpg", frame, cam,
                         [(ball_est, (0, 255, 255), "ball, as seen"), (S["face"], (255, 128, 0), "paddle centre")])
                fc, nrm = face(world, arm.q())
                S["truth"] = fc + nrm * BALL_R
                S["phase"], t_state = "correct", time.monotonic()
                say("Now rest the ball on the centre of the paddle, and hold it there.")
                w.hist.clear()
                w.last = None
            elif phase == "correct":
                truth = S["truth"]
                (uv,), _ = cam.project(truth[None])
                seen(w(frame, t, near=(uv[0], uv[1], cam.radius_px(truth))), t)
                held = w.still()
                if held is None:
                    if time.monotonic() - t_state > 45:
                        say("Skipping this one.", wait=True)
                        S["phase"] = "returning"
                    continue
                sight = cam.ball_in_world(*held)
                err = sight - truth
                pts.append({"uv": held[:2], "r": held[2], "truth": truth.tolist(), "seen": sight.tolist(),
                            "first_estimate": ball_est.tolist(), "face_target": plan_t.tolist(),
                            "q": arm.q().tolist()})
                (out / "points.json").write_text(json.dumps(pts, indent=1))
                snapshot(out / f"{len(pts) - 1:02d}_truth.jpg", frame, cam,
                         [(truth, (0, 255, 0), "true"), (sight, (0, 0, 255), "camera")])
                S["rounds"] = len(pts)
                S["last_err"] = (f"{np.linalg.norm(err) * 100:.1f} cm  (x {err[0] * 100:+.1f}, y {err[1] * 100:+.1f}, "
                                 f"z {err[2] * 100:+.1f})")
                print(f"#{len(pts)}  truth {np.round(truth, 3)}  camera {np.round(sight, 3)}  "
                      f"error {np.round(err * 100, 1)} cm", flush=True)
                say(f"Recorded. The camera was {np.linalg.norm(err) * 100:.0f} centimetres off. "
                    "Lift the ball away, and pick a new spot.", wait=True)
                S["phase"] = "returning"
            elif phase == "returning":
                arm.move_to(ready, speed=0.5)
                set_face()
                S["count_end"], S["phase"], last_said = time.monotonic() + args.countdown, "count", None
                w.hist.clear()
                w.last = None
                w.det.reset()
                n = camera.read()[2]
            elif phase == "count":
                seen(w(frame, t), t)  # the live readout keeps going while you choose
                left = int(np.ceil(S["count_end"] - time.monotonic()))
                if left != last_said and left > 0:
                    say(str(left))
                    last_said = left
                if left <= 0:
                    say("Hold it still now.")
                    w.hist.clear()
                    S["phase"] = "hold"
    except KeyboardInterrupt:
        pass
    finally:
        # The arm first: on Ctrl-C the terminal's pipes may already be gone,
        # and a failed print must not leave it standing with torque on.
        S["phase"] = "done"
        try:
            arm.hold()
            arm.fold()
        finally:
            arm.torque(False)
            painter.stop()
            camera.close()
            viewer.close()
        say("Done. The arm is folded.")
    (out / "points.json").write_text(json.dumps(pts, indent=1))
    return pts, out


def since_calibration() -> list[dict]:
    """Every correction from the sessions run since the camera was last
    calibrated with the arm - the ones that saw it where it is now."""
    from .cam_calibrate import OUT as CAL_OUT
    t_cal = (CAL_OUT / "points.json").stat().st_mtime
    pts = []
    for d in sorted(OUT.iterdir()):
        f = d / "points.json"
        if f.exists() and d.stat().st_mtime > t_cal:
            pts += json.loads(f.read_text())
    return pts


def refit(pts: list[dict], cam: CameraModel) -> tuple[CameraModel, dict]:
    """Camera placement, focal length and radius scale from the corrections,
    with the blade sightings from hw.cam_calibrate holding it in place where
    the corrections are few."""
    from scipy.optimize import least_squares

    from .cam_calibrate import OUT as CAL_OUT
    from .vision import PADDLE_R

    T = np.array([p["truth"] for p in pts])
    uv = np.array([p["uv"] for p in pts])
    r = np.array([p["r"] for p in pts])
    blade = json.loads((CAL_OUT / "points.json").read_text())["points"]
    B = np.array([b["p"] for b in blade])
    buv = np.array([b["uv"] for b in blade])
    bmaj = np.array([b["major"] for b in blade])

    def model(x):
        return CameraModel(f=x[0], cx=cam.cx, cy=cam.cy, R=cv2.Rodrigues(x[1:4])[0], t=x[4:7],
                           width=cam.width, height=cam.height, radius_scale=float(np.exp(x[7])))

    def residuals(x):
        m = model(x)
        pu, _ = m.project(T)
        rr = np.array([m.radius_px(p) for p in T])
        bu, _ = m.project(B)
        bd = np.linalg.norm(m.to_cam(B), axis=1)
        return np.r_[(pu - uv).ravel(), 2.0 * (rr - r),  # the ball: where, and how big
                     0.5 * (bu - buv).ravel(), 0.15 * (bmaj - 2 * PADDLE_R * m.f / bd)]  # the blade

    x0 = np.r_[cam.f, cv2.Rodrigues(cam.R)[0].ravel(), cam.t, np.log(cam.radius_scale)]
    x = least_squares(residuals, x0, loss="soft_l1", f_scale=5.0).x
    m = model(x)

    def err3d(c):
        e = np.array([c.ball_in_world(u, v, rad) for (u, v), rad in zip(uv, r)]) - T
        return np.sqrt((e**2).mean(axis=0)) * 100, float(np.sqrt((e**2).sum(axis=1).mean()) * 100)

    (bx, btot), (ax, atot) = err3d(cam), err3d(m)
    stats = {"points": len(pts), "before_cm_rms_xyz": bx.round(1).tolist(), "before_cm_rms": round(btot, 1),
             "after_cm_rms_xyz": ax.round(1).tolist(), "after_cm_rms": round(atot, 1),
             "f": round(m.f, 1), "radius_scale": round(m.radius_scale, 3),
             "camera_moved_cm": round(float(np.linalg.norm(m.position - cam.position)) * 100, 1)}
    return m, stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--points", type=int, default=None)
    p.add_argument("--countdown", type=int, default=10, help="seconds counted out between rounds")
    p.add_argument("--port", type=int, default=8765, help="live view at http://localhost:<port>")
    p.add_argument("--no-browser", action="store_true", help="do not open the live view")
    p.add_argument("--ball", choices=("white", "orange"), default="white")
    p.add_argument("--refit", action="store_true", help="refit from the latest session's points")
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args(argv)
    if args.refit:
        out = sorted(d for d in OUT.iterdir() if (d / "points.json").exists())[-1]
    else:
        _, out = session(args)
    pts = since_calibration()
    if len(pts) < 3:
        print(f"{len(pts)} corrections; need 3 or more to refit", flush=True)
        return 0
    cam = CameraModel.load()
    m, stats = refit(pts, cam)
    text = json.dumps(stats, indent=1)
    print(text, flush=True)
    (out / "report.txt").write_text(text + "\n")
    if not args.no_write and stats["after_cm_rms"] < stats["before_cm_rms"]:
        shutil.copy(CAMERA_FILE, out / "camera_before.json")
        m.save(CAMERA_FILE, fit={"from": "hw.ball_point", **stats})
        print("wrote", CAMERA_FILE, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
