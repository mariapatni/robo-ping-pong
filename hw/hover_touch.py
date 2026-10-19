"""Hover-the-ball touch test: does the paddle meet the ball where the camera says it is?

    .venv/bin/python -m hw.hover_touch --sim                 20 spots, scored against the sim's truth
    .venv/bin/python -m hw.hover_touch --sim --n 5 --seed 3

The ball is held still somewhere over the table. The camera finds it and puts
its centre B in the world frame (the robot's: origin at the base, z up). The
paddle then comes up underneath, face flat - its normal along +z - with the
centre of the face on the vertical line through B, and climbs until it
touches. A flat face under a sphere touches it at exactly one point, the
bottom of the ball, straight below the centre; so where the paddle is when it
first touches says, with no ambiguity, where the ball really is:

    sideways    the face centre's x, y at first touch  minus  the ball centre's x, y
    height      the face's z at first touch            minus  (B_z - 20 mm)

and the plan, made from the camera alone, said the touch would come at
z_plan = B_cam_z - 20 mm. Everything is in the one world frame, along its z.

Before the first spot, the ball's apparent size is calibrated (`radius_scale`):
the ball rests on the table at a few places, where its centre is exactly 20 mm
up, so the camera ray through it gives its distance without its size. Depth
from one camera comes from the ball's size alone, and the detector reads the
rendered outline ~0.3 px small - mixed pixels at the edge - which is 1% of the
radius and 5 mm of depth at half a metre.

THE SIM (this file, `--sim`)
  The ball is a fixed orange sphere (a mocap body), so contact is the
  physics' own and its true centre is known exactly. The camera is a 1280 x 720
  pinhole like the FaceTime camera, placed where the laptop sits; the ball is
  found in its rendered frame by the same detector the real tracker runs
  (hw.vision, orange). The arm is parked folded while the camera looks, as on
  the real arm. The approach, planned against the camera's estimate of the
  ball (the truth is never used to plan):
    0. out of the folded pose and under the ball with the face flat, low, by
       waypoints whose joint-space paths keep every part of the arm 12 mm
       clear of the estimated ball;
    1. 40 mm below the planned touch, face flat;
    2. 10 mm below it;
    3. up 1 mm at a time until the paddle touches, at most 25 mm past the plan.
  Each move is solved with sim.pose's IK on the face, then corrected by what
  the joints actually reached (the servos sag under gravity - in the sim as on
  the arm, where the encoders give the same correction).

THE REAL ARM (not yet)
  The same plan and the same climb; contact is the ball, held lightly, being
  lifted by more than 1 mm in the image, and the sideways error is the ball's
  centre against the STag centre on the paddle in the same frame.

Pass: every axis within 10 mm at every spot.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np

from hw.vision import BALL_R, BallDetector, CameraModel, DetectorConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "hover_touch"

PRE = 0.040  # first stop, below the planned touch
START = 0.010  # where the climb starts
STEP = 0.001  # climb step
OVERSHOOT = 0.025  # give up this far past the planned touch
PASS = 0.010
LOW = 0.05  # face height for passing under the ball
CLEAR = 0.012  # nothing of the arm comes nearer the (estimated) ball than this on the way in

# Where the ball is held, in the world frame (m). The face has to get under it
# flat, which the 5-joint arm can do over roughly this box.
SPOTS = ((0.17, 0.34), (-0.15, 0.15), (0.16, 0.30))

# The sim camera: a FaceTime-like 1280 x 720 pinhole (70 deg across), where the
# laptop sits - to the arm's right and in front of it, looking at the workspace.
SIZE = (1280, 720)
CAM_K = np.array([[905.0, 0.0, 640.0], [0.0, 905.0, 360.0], [0.0, 0.0, 1.0]])
CAM_POS = np.array([0.60, -0.40, 0.32])
CAM_LOOK = np.array([0.25, 0.0, 0.18])

# hw.so101's fold, turned 70 degrees off to the side: folded straight ahead,
# the 150 mm blade stands up in the middle of the camera's view and hides balls
# behind it.
FOLD = np.array([1.2, -1.10, 1.64, -1.55, 0.0])
BALL_RGBA = (1.0, 0.42, 0.06, 1.0)


@dataclass
class Trial:
    spot: int
    truth: list[float]  # ball centre, m
    seen: list[float] | None  # the camera's estimate, m
    plan_z: float | None  # where the face should touch, m
    touched: bool
    face_at_touch: list[float] | None  # face centre at first contact, m
    tilt_deg: float | None  # face normal from +z at first contact
    xy_err_mm: list[float] | None  # face centre - ball centre, x and y
    z_err_mm: float | None  # plan - face z at touch: how far off the camera put the ball's bottom
    contact_err_mm: float | None  # face z at touch - (ball z - R): ~0 if it met the bottom, flat
    cam_err_mm: list[float] | None  # camera's estimate - truth
    contact_off_mm: float | None  # contact point from the face centre, sideways
    note: str = ""

    @property
    def passed(self) -> bool:
        return bool(self.touched and max(np.abs(self.xy_err_mm).max(), abs(self.z_err_mm)) <= PASS * 1000)


# -- the sim world ------------------------------------------------------------


class SimRig:
    """The arm with the tagged paddle, the table, a hovering ball and the camera."""

    def __init__(self):
        from sim.arm import Arm
        from sim.cameras import add_cv_camera, look_at
        from sim.config import EnvConfig
        from sim.paddle import FACE_SITE, PADDLE_BODY, add_paddle, tagged_paddle
        from sim.safety import TableGuard
        from sim.table import add_corner_markers, add_table

        self.cfg = EnvConfig(paddle=tagged_paddle())
        R = look_at(CAM_POS, CAM_LOOK)
        self.cam = CameraModel(f=CAM_K[0, 0], cx=CAM_K[0, 2], cy=CAM_K[1, 2], R=R, t=-R @ CAM_POS,
                               width=SIZE[0], height=SIZE[1])

        def build(spec):
            spec.option.timestep = self.cfg.timestep
            spec.visual.quality.offsamples = 8
            add_table(spec, self.cfg.table, self.cfg.surface)
            add_corner_markers(spec, self.cfg.table, self.cfg.markers)
            add_paddle(spec, self.cfg.paddle)
            ball = spec.worldbody.add_body(name="hover_ball", mocap=True, pos=[0.3, 0.0, 0.3])
            ball.add_geom(name="hover_ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_R, 0, 0],
                          rgba=list(BALL_RGBA), condim=3)
            # Where the camera thinks the ball is: what the path planner keeps clear of.
            ghost = spec.worldbody.add_body(name="ghost_ball", mocap=True, pos=[0.3, 0.0, 0.3])
            ghost.add_geom(name="ghost_ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_R, 0, 0],
                           rgba=[1, 0, 1, 0], contype=0, conaffinity=0, group=4)
            add_cv_camera(spec, "laptop", CAM_K, R, -R @ CAM_POS, SIZE)

        self.arm = Arm(offscreen=SIZE, modify=build)
        m = self.arm.model
        self.face = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, FACE_SITE)
        self.paddle = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PADDLE_BODY)
        self.ball_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "hover_ball")
        self.mocap = int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hover_ball")])
        self.ghost_mocap = int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ghost_ball")])
        self.ghost = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ghost_ball")
        # Every colliding part of the arm, paddle included: what has to stay clear of the ball.
        skip = {0, int(m.geom_bodyid[self.ball_geom]), int(m.geom_bodyid[self.ghost])}
        self.arm_geoms = [g for g in range(m.ngeom)
                          if m.geom_bodyid[g] not in skip and (m.geom_contype[g] or m.geom_conaffinity[g])]
        self.guard = TableGuard(self.arm)
        self.renderer = mujoco.Renderer(m, SIZE[1], SIZE[0])
        self._fk = mujoco.MjData(m)
        self.live = None  # a lab.live.Live, to stream the camera's view while the arm moves
        self.caption = ""

    def place_ball(self, p) -> None:
        self.arm.data.mocap_pos[self.mocap] = p
        mujoco.mj_forward(self.arm.model, self.arm.data)

    def believe_ball(self, p) -> None:
        """Where the planner thinks the ball is."""
        self.arm.data.mocap_pos[self.ghost_mocap] = p
        self._fk.mocap_pos[self.ghost_mocap] = p
        mujoco.mj_forward(self.arm.model, self.arm.data)

    def clearance(self, q) -> float:
        """Nearest approach of any part of the arm, at q, to the believed ball (m)."""
        d = self._fk
        d.qpos[self.arm.qpos_adr] = q
        mujoco.mj_forward(self.arm.model, d)
        ft = np.zeros(6)
        return min(mujoco.mj_geomDistance(self.arm.model, d, g, self.ghost, 0.2, ft) for g in self.arm_geoms)

    def path_clear(self, q0, q1, n: int = 24) -> bool:
        return all(self.clearance(q) >= CLEAR and self.guard.is_safe(q)
                   for q in (q0 + a * (q1 - q0) for a in np.linspace(0, 1, n)))

    def plan_in(self, under) -> list[np.ndarray] | None:
        """Joint waypoints from where the arm is to flat, PRE below `under`,
        every leg clear of the believed ball; None if there is no such way."""
        pre = np.asarray(under) - [0, 0, PRE]
        low = np.array([pre[0], pre[1], min(LOW, pre[2])])
        here = self.face_pose()[0]
        for legs in ([pre], [low, pre], [np.array([here[0], here[1], LOW]), low, pre]):
            q, qs = self.arm.qpos, []
            for w in legs:
                s = self.solve(w, q)
                if s.position_error > 0.003 or s.normal_error_deg > 3 or not self.path_clear(q, s.qpos):
                    break
                qs.append(s.qpos)
                q = s.qpos
            else:
                return qs
        return None

    def follow(self, qs) -> np.ndarray | None:
        """Drive through joint waypoints along straight joint-space legs.
        Returns the contact if anything touched the ball on the way."""
        q = self.arm.qpos
        for q1 in qs:
            for a in np.linspace(0, 1, 13)[1:]:
                hit = self.move(q + a * (q1 - q), timeout=0.4, stop_on_touch=True)
                if hit is not None:
                    return hit
            q = q1
        return self.move(q, stop_on_touch=True)

    def frame(self) -> np.ndarray:
        self.renderer.update_scene(self.arm.data, "laptop")
        return cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)

    def face_pose(self, q=None) -> tuple[np.ndarray, np.ndarray]:
        """Face centre and normal, at `q` or where the arm is now."""
        d = self.arm.data if q is None else self._fk
        if q is not None:
            d.qpos[self.arm.qpos_adr] = q
            mujoco.mj_kinematics(self.arm.model, d)
        return d.site_xpos[self.face].copy(), d.site_xmat[self.face].reshape(3, 3)[:, 2].copy()

    def touching(self) -> np.ndarray | None:
        """The paddle-ball contact point, if they touch."""
        m, d = self.arm.model, self.arm.data
        for c in d.contact[: d.ncon]:
            g = {int(c.geom1), int(c.geom2)}
            if self.ball_geom in g and any(m.geom_bodyid[k] == self.paddle for k in g - {self.ball_geom}):
                return c.pos.copy()
        return None

    def solve(self, target, q0):
        from sim.pose import solve_face_pose_ik
        return solve_face_pose_ik(self.arm, target, (0.0, 0.0, 1.0), site=self.face, q0=q0, restarts=2,
                                  iters=250, guard=self.guard, seed=0)

    def move(self, q, timeout: float = 2.0, stop_on_touch: bool = False):
        """Drive to joint target q until it settles; with `stop_on_touch`,
        stop the moment the paddle touches the ball and return the contact."""
        arm = self.arm
        arm.command(q)
        end = arm.data.time + timeout
        while arm.data.time < end:
            arm.step(arm.steps_per_command)
            if self.live is not None and self.live.due():
                img = self.frame()
                cv2.putText(img, self.caption, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                self.live.frame(img)
            hit = self.touching()
            if stop_on_touch and hit is not None:
                arm.command(arm.qpos)  # hold where it is
                return hit
            if np.abs(arm.qpos - q).max() < 2e-3 and np.abs(arm.qvel).max() < 1e-2:
                break
        return None

    def reach(self, target, stop_on_touch: bool = False, rounds: int = 3):
        """Put the face centre at `target`, flat, correcting for where the
        joints actually settle. Returns (contact or None, IK solution)."""
        want = np.asarray(target, float)
        aim, sol = want.copy(), None
        for _ in range(rounds):
            sol = self.solve(aim, self.arm.qpos)
            hit = self.move(sol.qpos, stop_on_touch=stop_on_touch)
            if hit is not None:
                return hit, sol
            err = self.face_pose()[0] - want
            if np.linalg.norm(err) < 3e-4:
                break
            aim = aim - err
        return None, sol


# -- the experiment -----------------------------------------------------------


def find_ball(rig: SimRig, det: BallDetector) -> tuple[np.ndarray | None, np.ndarray, object]:
    img = rig.frame()
    d = None
    for _ in range(3):  # the detector settles its white balance over a few frames
        d = det(img, roi=rig.cam.roi())
    if d is None:
        return None, img, None
    return rig.cam.ball_in_world(d.u, d.v, d.r), img, d


def calibrate_radius(rig: SimRig, det: BallDetector, n: int = 6, seed: int = 0) -> float:
    """The detector's radius times this is the true image radius: from the
    ball resting on the table (centre at z = R), where the ray alone fixes it."""
    rng = np.random.default_rng(seed)
    rig.arm.reset(FOLD)
    cam = rig.cam
    cam.radius_scale = 1.0
    ratios = []
    while len(ratios) < n:
        p = np.array([rng.uniform(0.15, 0.45), rng.uniform(-0.2, 0.2), BALL_R])
        rig.place_ball(p)
        det.reset()
        img = rig.frame()
        for _ in range(3):
            d = det(img, roi=cam.roi())
        if d is None:
            continue
        # The ray through the detected centre, met with the plane z = R...
        ray = cam.R.T @ np.array([(d.u - cam.cx) / cam.f, (d.v - cam.cy) / cam.f, 1.0])
        o = cam.position
        on_table = o + ray * (BALL_R - o[2]) / ray[2]
        # ...is where the ball is, and so how big it ought to look.
        ratios.append(cam.radius_px(on_table) / d.r)
    cam.radius_scale = float(np.median(ratios))
    return cam.radius_scale


def reachable(rig: SimRig, p) -> bool:
    """Can the face get flat under a ball at p from the fold, and reach the touch?"""
    rig.arm.reset(FOLD)
    rig.believe_ball(p)
    s = rig.solve(np.asarray(p) - [0, 0, BALL_R], FOLD)
    if s.position_error > 0.003 or s.normal_error_deg > 3 or not rig.guard.is_safe(s.qpos):
        return False
    return rig.plan_in(np.asarray(p) - [0, 0, BALL_R]) is not None


def trial(rig: SimRig, k: int, truth: np.ndarray, det: BallDetector, snap: Path | None) -> Trial:
    arm = rig.arm
    arm.reset(FOLD)
    rig.place_ball(truth)
    det.reset()
    seen, img, d = find_ball(rig, det)
    t = Trial(k, truth.tolist(), None, None, False, None, None, None, None, None, None, None)
    if seen is None:
        t.note = "the camera did not find the ball"
        return t
    t.seen, t.cam_err_mm = seen.tolist(), ((seen - truth) * 1000).tolist()
    plan_z = float(seen[2] - BALL_R)
    t.plan_z = plan_z
    under = np.array([seen[0], seen[1], plan_z])

    rig.believe_ball(seen)
    path = rig.plan_in(under)
    if path is None:
        t.note = "no way in that clears the ball"
        return t
    if rig.follow(path) is not None or rig.reach(under - [0, 0, PRE], stop_on_touch=True)[0] is not None:
        t.note = "hit the ball on the way in"
        return t
    hit, _ = rig.reach(under - [0, 0, START], stop_on_touch=True)
    if hit is not None:
        t.note = "touched before the climb"
    z = plan_z - START
    while hit is None and z < plan_z + OVERSHOOT:
        z += STEP
        hit, _ = rig.reach([under[0], under[1], z], stop_on_touch=True, rounds=2)
    if hit is None:
        t.note = f"no touch {OVERSHOOT * 1000:.0f} mm past the plan"
        return t

    face, normal = rig.face_pose()
    t.touched = True
    t.face_at_touch = face.tolist()
    t.tilt_deg = float(np.degrees(np.arccos(np.clip(normal[2], -1, 1))))
    t.xy_err_mm = ((face[:2] - truth[:2]) * 1000).tolist()
    t.z_err_mm = float((plan_z - face[2]) * 1000)
    t.contact_err_mm = float((face[2] - (truth[2] - BALL_R)) * 1000)
    t.contact_off_mm = float(np.linalg.norm(hit[:2] - face[:2]) * 1000)
    if snap is not None:
        draw(rig, img, d, truth, seen, snap / f"spot{k:02d}_seen.png")
        touch = rig.frame()
        draw(rig, touch, None, truth, seen, snap / f"spot{k:02d}_touch.png", face=face, t=t)
    return t


def draw(rig: SimRig, img, d, truth, seen, path: Path, face=None, t: Trial | None = None) -> None:
    img = img.copy()
    cam = rig.cam
    if d is not None:
        cv2.circle(img, (int(round(d.u)), int(round(d.v))), int(round(d.r)), (255, 255, 0), 1, cv2.LINE_AA)

    def mark(p, col, label, kind=cv2.MARKER_CROSS):
        (uv,), _ = cam.project(np.asarray(p)[None])
        cv2.drawMarker(img, (int(uv[0]), int(uv[1])), col, kind, 18, 2)
        cv2.putText(img, label, (int(uv[0]) + 12, int(uv[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1,
                    cv2.LINE_AA)

    mark(truth, (0, 220, 0), "truth")
    mark(seen, (255, 0, 255), "camera", cv2.MARKER_TILTED_CROSS)
    # The vertical through the camera's estimate, down to the table.
    (a, b), _ = cam.project(np.array([seen, [seen[0], seen[1], 0.0]]))
    cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), (255, 0, 255), 1, cv2.LINE_AA)
    if face is not None:
        mark(face, (0, 160, 255), "face centre", cv2.MARKER_DIAMOND)
    lines = [f"truth {np.round(np.array(truth) * 1000).astype(int)} mm  camera {np.round(np.array(seen) * 1000).astype(int)} mm"]
    if t is not None and t.touched:
        lines.append(f"sideways {t.xy_err_mm[0]:+.1f}, {t.xy_err_mm[1]:+.1f} mm   height {t.z_err_mm:+.1f} mm   "
                     f"tilt {t.tilt_deg:.1f} deg   {'PASS' if t.passed else 'FAIL'}")
    cv2.rectangle(img, (0, 0), (img.shape[1], 22 * len(lines) + 10), (20, 20, 20), -1)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def summarise(trials: list[Trial]) -> dict:
    hit = [t for t in trials if t.touched]
    if not hit:
        return {"spots": len(trials), "touched": 0, "passed": 0}
    xy = np.array([t.xy_err_mm for t in hit])
    z = np.array([t.z_err_mm for t in hit])
    cam = np.array([t.cam_err_mm for t in hit])

    def rms(a):
        return float(np.sqrt(np.mean(np.square(a))))

    return {
        "spots": len(trials), "touched": len(hit), "passed": sum(t.passed for t in trials),
        "sideways_rms_mm": rms(np.linalg.norm(xy, axis=1)), "sideways_max_mm": float(np.linalg.norm(xy, axis=1).max()),
        "height_rms_mm": rms(z), "height_max_mm": float(np.abs(z).max()), "height_mean_mm": float(z.mean()),
        "camera_rms_mm": [rms(cam[:, i]) for i in range(3)],
        "tilt_max_deg": float(max(t.tilt_deg for t in hit)),
        "contact_err_max_mm": float(max(abs(t.contact_err_mm) for t in hit)),
    }


def run_sim(n: int, seed: int, out: Path | None, log=print) -> tuple[list[Trial], dict]:
    from lab.live import Live
    from lab.experiments import EXPERIMENTS
    rig = SimRig()
    live = rig.live = Live("hover_touch", EXPERIMENTS["hover_touch"]["title"], EXPERIMENTS["hover_touch"]["description"],
                           kind="calibration")
    det = BallDetector(DetectorConfig(ball="orange"))
    scale = calibrate_radius(rig, det, seed=seed + 1000)
    log(f"radius_scale {scale:.4f}, from the ball resting on the table")
    rng = np.random.default_rng(seed)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    trials = []
    tries = 0
    while len(trials) < n and tries < 50 * n:
        tries += 1
        p = np.array([rng.uniform(*SPOTS[0]), rng.uniform(*SPOTS[1]), rng.uniform(*SPOTS[2])])
        if not reachable(rig, p):
            continue
        rig.caption = f"spot {len(trials) + 1}/{n}: ball at {np.round(p * 1000).astype(int)} mm"
        live.status(spot=f"{len(trials) + 1}/{n}")
        t = trial(rig, len(trials), p, det, out)
        trials.append(t)
        if t.touched:
            live.point("sideways mm", t.spot, float(np.hypot(*t.xy_err_mm)))
            live.point("height mm", t.spot, abs(t.z_err_mm))
            live.status(passed=sum(x.passed for x in trials), last_sideways_mm=round(float(np.hypot(*t.xy_err_mm)), 2),
                        last_height_mm=round(t.z_err_mm, 2))
            live.log(f"spot {t.spot}: sideways {t.xy_err_mm[0]:+.1f} {t.xy_err_mm[1]:+.1f} mm, height {t.z_err_mm:+.1f} mm"
                     f" {'PASS' if t.passed else 'FAIL'}")
        else:
            live.log(f"spot {t.spot}: {t.note}")
        if t.touched:
            log(f"spot {t.spot:2d}  ball {np.round(p * 1000).astype(int)} mm   camera off "
                f"{np.round(t.cam_err_mm, 1)} mm   sideways {t.xy_err_mm[0]:+5.1f} {t.xy_err_mm[1]:+5.1f} mm   "
                f"height {t.z_err_mm:+5.1f} mm   tilt {t.tilt_deg:.1f}  {'PASS' if t.passed else 'FAIL'} {t.note}")
        else:
            log(f"spot {t.spot:2d}  ball {np.round(p * 1000).astype(int)} mm   {t.note}")
    s = summarise(trials)
    s["radius_scale"] = scale
    live.done(**{k: v for k, v in s.items() if not isinstance(v, list)})
    if out is not None:
        (out / "results.json").write_text(json.dumps({"summary": s, "trials": [asdict(t) for t in trials]},
                                                     indent=2) + "\n")
    return trials, s


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sim", action="store_true", help="run in the sim, scored against its truth")
    p.add_argument("--n", type=int, default=20, help="how many spots")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    if not a.sim:
        p.error("only --sim exists so far; the real-arm version follows the hand-eye calibration")
    out = a.out or OUT / time.strftime("sim_%Y%m%d_%H%M%S")
    _, s = run_sim(a.n, a.seed, out)
    print(json.dumps(s, indent=2))
    print(f"wrote {out}")
    return 0 if s.get("passed") == s.get("spots") else 1


if __name__ == "__main__":
    raise SystemExit(main())
