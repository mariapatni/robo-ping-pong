"""Hand-eye calibration: where the fixed camera is, from the STag markers on the paddle.

    .venv/bin/python -m hw.handeye --sim                  the proof: rendered views, scored against truth
    .venv/bin/python -m hw.handeye --sim --no-render      the same fit on exact corners + 0.15 px noise
    .venv/bin/python -m hw.handeye --sim --seed 3 --poses 40 --held 12

The camera does not move; the arm carries the markers - STag HD21 id 0 on the
hitting face, id 1 on the back (sim.paddle.tagged_paddle). For each of ~40
poses the arm stops, the camera finds a marker's four corners, and the arm's
kinematics say where those corners are. One least-squares fit over every
corner of every pose then solves, together:

    the camera's pose in the world (the robot's frame: base at the origin, z up)   6
    offsets on shoulder_lift, elbow_flex, wrist_flex, wrist_roll                    4
    how the paddle sits on the wrist: 3 mm-scale shifts, 2 small tilts              5

with the lens (K) fixed from the ChArUco calibration. Two things it cannot
separate, by geometry, and so does not try:
  - a shoulder_pan offset is the camera turning about the base's z axis; the
    world frame's x is defined by pan = 0, so the camera absorbs it. (A marker
    on the base, id 2, would pin it.)
  - the paddle turned on the wrist about the roll axis is a wrist_roll offset.
    The fit puts it all in the roll offset; the kinematics are identical.

THE PROOF (`--sim`)
  A truth the fit never sees: a random camera near where the laptop sits, a
  lens whose calibration is slightly off (focal +0.1%, centre 1.5 px - the
  residuals the ChArUco proof leaves), the elbow 17 deg off and the other three
  joints up to 3 deg off, and the paddle mounted 2 mm and 2 deg wrong. Poses
  are planned with only a rough guess of the camera (5 cm, 5 deg off), so some
  views miss, as they would. Each view is rendered and read by the real STag
  detector. Then, against the truth:
    camera position and angle error
    each joint offset and the mount error
    reprojection error, per image, on the fit views and on held-out views
    3D: marker corners and the paddle face centre, predicted against true, on
        held-out views and on 300 random poses across the workspace (RMSE)
  Pass: camera within 2 mm and 0.1 deg, offsets within 0.2 deg, fit within
  0.4 px, and the paddle face within 1 cm everywhere (aim: a few mm).

Results go to out/handeye/<run>/: results.json, log.txt, and overlays of a few
views (detected corners, and where the fit puts them).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "handeye"

SIZE = (1280, 720)
FIT_JOINTS = (1, 2, 3, 4)  # lift, elbow, wrist flex, wrist roll: pan is the camera's
MAX_VIEW_DEG = 60.0  # marker normal to camera ray
MIN_EDGE_PX = 45.0
MARGIN_PX = 40.0
EDGE_PX = 12.0  # a detection with a corner nearer the frame edge than this is dropped
OUTLIER = 3.0  # a view fitting worse than this many times the median view is dropped, then refit
OUTLIER_FLOOR_PX = 1.0

# The laptop camera's neighbourhood (hw.hover_touch's camera is the centre).
CAM_POS = np.array([0.60, -0.40, 0.32])
CAM_LOOK = np.array([0.25, 0.0, 0.18])

PASS = {"cam_mm": 2.0, "cam_deg": 0.1, "offset_deg": 0.2, "reproj_px": 0.4, "face_mm": 10.0}


# -- small rotation helpers ---------------------------------------------------


def rot(rv) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(rv, float).reshape(3, 1))[0]


def angle_deg(R) -> float:
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(np.asarray(R, float))[0])))


# -- the truth (sim only) -----------------------------------------------------


@dataclass
class Truth:
    K: np.ndarray  # the real lens
    R: np.ndarray  # world -> camera
    t: np.ndarray
    offsets: np.ndarray  # (5,) rad: true angle = reported + offset
    mount_t: np.ndarray  # paddle mount error, in the paddle frame (m)
    mount_r: np.ndarray  # rotation vector (rad), in the paddle frame

    @property
    def cam_pos(self) -> np.ndarray:
        return -self.R.T @ self.t


def make_truth(rng) -> Truth:
    from sim.cameras import look_at
    pos = CAM_POS + rng.uniform(-0.08, 0.08, 3)
    R = look_at(pos, CAM_LOOK + rng.uniform(-0.05, 0.05, 3))
    # A roll of a few degrees: a laptop lid is not perfectly level.
    R = rot([0, 0, np.radians(rng.uniform(-4, 4))]) @ R
    f = rng.uniform(880, 930)
    K = np.array([[f, 0, 640 + rng.uniform(-8, 8)], [0, f * rng.uniform(0.998, 1.002), 360 + rng.uniform(-8, 8)],
                  [0, 0, 1]])
    off = np.zeros(5)
    off[1:] = np.radians(rng.uniform(-3, 3, 4))
    off[2] = np.radians(17.0 * rng.choice([-1, 1]))  # the elbow, as found on the real arm
    mt = rng.normal(size=3)
    mt *= 0.002 / np.linalg.norm(mt)
    # Tilt only about the paddle's y and z: about x (the roll axis) it is a roll offset.
    mr = np.r_[0.0, rng.normal(size=2)]
    mr *= np.radians(2.0) / np.linalg.norm(mr)
    return Truth(K, R, -R @ pos, off, mt, mr)


def lens_estimate(K: np.ndarray, exact: bool = False) -> np.ndarray:
    """What the ChArUco calibration hands over: the truth, off by about what
    the sim lens proof measured (hw/sim_intrinsics.py)."""
    Ke = K.copy()
    if exact:
        return Ke
    Ke[0, 0] *= 1.001
    Ke[1, 1] *= 1.001
    Ke[0, 2] += 1.5
    Ke[1, 2] -= 1.5
    return Ke


# -- the arm model, as the fit sees it ----------------------------------------


class Kinematics:
    """Paddle pose from reported joint angles, nominal model plus the fitted
    corrections."""

    def __init__(self, model: mujoco.MjModel, qpos_adr, paddle_body: int):
        self.model, self.adr, self.body = model, qpos_adr, paddle_body
        self.data = mujoco.MjData(model)

    def paddle(self, q) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[self.adr] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.xmat[self.body].reshape(3, 3).copy(), self.data.xpos[self.body].copy()

    def points(self, q, local, offsets, mount_t, mount_R) -> np.ndarray:
        """World positions of paddle-frame points `local` (n, 3)."""
        Rb, pb = self.paddle(np.asarray(q, float) + offsets)
        return (local @ mount_R.T + mount_t) @ Rb.T + pb


@dataclass
class View:
    q: np.ndarray  # reported joint angles
    tag: int
    corners: np.ndarray  # (4, 2) detected, STag order
    held: bool = False
    image: np.ndarray | None = field(default=None, repr=False)


# -- the sim world --------------------------------------------------------------


class SimWorld:
    def __init__(self, truth: Truth, render: bool = True):
        from sim.arm import Arm
        from sim.cameras import add_cv_camera
        from sim.paddle import PADDLE_BODY, add_paddle, tagged_paddle
        from sim.safety import TableGuard
        from sim.config import EnvConfig
        from sim.table import add_corner_markers, add_table

        self.truth = truth
        self.cfg = EnvConfig(paddle=tagged_paddle())
        self.tags = {g.id: g for g in self.cfg.paddle.tags}

        def build(spec, wrong: bool):
            spec.visual.quality.offsamples = 8
            add_table(spec, self.cfg.table, self.cfg.surface)
            add_corner_markers(spec, self.cfg.table, self.cfg.markers)
            add_paddle(spec, self.cfg.paddle)
            if wrong:  # the paddle as it is really bolted on
                b = spec.body(PADDLE_BODY)
                R0 = np.zeros(9)
                mujoco.mju_quat2Mat(R0, np.asarray(b.quat, float))
                R0 = R0.reshape(3, 3)
                b.pos = (np.asarray(b.pos) + R0 @ truth.mount_t).tolist()
                q = np.zeros(4)
                mujoco.mju_mat2Quat(q, (R0 @ rot(truth.mount_r)).reshape(9))
                b.quat = q.tolist()
                add_cv_camera(spec, "cam", truth.K, truth.R, truth.t, SIZE)

        self.real = Arm(offscreen=SIZE, modify=lambda s: build(s, True))  # what the camera sees
        self.nominal = Arm(modify=lambda s: build(s, False))  # what the CAD says
        m = self.nominal.model
        self.body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PADDLE_BODY)
        self.kin = Kinematics(m, self.nominal.qpos_adr, self.body)
        self.real_kin = Kinematics(self.real.model, self.real.qpos_adr,
                                   mujoco.mj_name2id(self.real.model, mujoco.mjtObj.mjOBJ_BODY, PADDLE_BODY))
        self.guard = TableGuard(self.nominal)
        self.renderer = mujoco.Renderer(self.real.model, SIZE[1], SIZE[0]) if render else None

    def true_points(self, q_reported, local) -> np.ndarray:
        Rb, pb = self.real_kin.paddle(np.asarray(q_reported) + self.truth.offsets)
        return local @ Rb.T + pb

    def capture(self, q_reported, rng, noise_px: float = 0.15) -> list[tuple[int, np.ndarray, np.ndarray | None]]:
        """(tag id, corners, image) for each marker the camera reads with the
        arm at q (reported). Rendered and detected, or projected plus noise."""
        if self.renderer is None:
            out = []
            for tid, tag in self.tags.items():
                P = self.true_points(q_reported, tag.corners())
                uv, z = project(self.truth.K, self.truth.R, self.truth.t, P)
                cam = -self.truth.R.T @ self.truth.t
                n = self.true_points(q_reported, np.array([tag.centre, np.asarray(tag.centre) + tag.outward]))
                facing = (n[1] - n[0]) @ (cam - n[0]) / np.linalg.norm(cam - n[0])
                if np.all(z > 0) and facing > np.cos(np.radians(75)) and inside(uv, 2):
                    out.append((tid, uv + rng.normal(0, noise_px, uv.shape), None))
            return out
        import stag
        d = self.real.data
        d.qpos[self.real.qpos_adr] = np.asarray(q_reported) + self.truth.offsets
        mujoco.mj_forward(self.real.model, d)
        self.renderer.update_scene(d, "cam")
        img = np.ascontiguousarray(self.renderer.render()[..., ::-1])
        corners, ids, _ = stag.detectMarkers(img, 21)
        if ids is None:
            return []
        # A marker running off the frame still decodes, with its lost corners
        # extrapolated: no use for a calibration.
        return [(int(i), c.reshape(4, 2).astype(float), img) for c, i in zip(corners, np.ravel(ids))
                if int(i) in self.tags and inside(c.reshape(4, 2), EDGE_PX)]


def project(K, R, t, P) -> tuple[np.ndarray, np.ndarray]:
    pc = np.atleast_2d(P) @ R.T + t
    return pc[:, :2] / pc[:, 2:3] * [K[0, 0], K[1, 1]] + K[:2, 2], pc[:, 2]


def inside(uv, margin) -> bool:
    return bool(np.all((uv[:, 0] > margin) & (uv[:, 0] < SIZE[0] - margin) &
                       (uv[:, 1] > margin) & (uv[:, 1] < SIZE[1] - margin)))


# -- choosing the poses ---------------------------------------------------------


def plan_poses(world: SimWorld, K, R, t, n: int, rng, pool: int = 4000) -> list[tuple[np.ndarray, int]]:
    """n arm poses that each show a marker to the camera (as far as a rough
    guess of the camera can tell): facing it within MAX_VIEW_DEG, corners
    MARGIN_PX inside the frame, edges at least MIN_EDGE_PX, the arm clear of
    the table. Spread by farthest-point sampling over where the marker is and
    which way it faces, so the fit sees every direction."""
    arm = world.nominal
    lo, hi = arm.lower.copy(), arm.upper.copy()
    lo[1], hi[1] = -1.2, 1.2  # keep the paddle over the table, not behind the base
    cam = -R.T @ t
    cands, feats = [], []
    tries = 0
    while len(cands) < pool // 4 and tries < pool * 6:
        tries += 1
        q = rng.uniform(lo, hi)
        if not world.guard.is_safe(q):
            continue
        Rb, pb = world.kin.paddle(q)
        for tid, tag in world.tags.items():
            c = Rb @ np.asarray(tag.centre) + pb
            nrm = Rb @ tag.outward
            ray = cam - c
            cos = nrm @ ray / np.linalg.norm(ray)
            if cos < np.cos(np.radians(MAX_VIEW_DEG)):
                continue
            uv, z = project(K, R, t, tag.corners() @ Rb.T + pb)
            if np.any(z <= 0.1) or not inside(uv, MARGIN_PX):
                continue
            if np.linalg.norm(np.roll(uv, -1, 0) - uv, axis=1).min() < MIN_EDGE_PX:
                continue
            cands.append((q, tid))
            feats.append(np.r_[c / 0.15, nrm, 0.5 * np.cos(q[4]), 0.5 * np.sin(q[4])])
    if len(cands) < n:
        raise RuntimeError(f"only {len(cands)} usable poses found")
    F = np.array(feats)
    pick = [int(rng.integers(len(F)))]
    d = np.linalg.norm(F - F[pick[0]], axis=1)
    while len(pick) < n:
        k = int(np.argmax(d))
        pick.append(k)
        d = np.minimum(d, np.linalg.norm(F - F[k], axis=1))
    return [cands[k] for k in pick]


# -- the fit --------------------------------------------------------------------


N_PARAMS = 6 + len(FIT_JOINTS) + 5


def unpack(x):
    rv, tv = x[:3], x[3:6]
    off = np.zeros(5)
    off[list(FIT_JOINTS)] = x[6:6 + len(FIT_JOINTS)]
    m = x[6 + len(FIT_JOINTS):]
    return rot(rv), tv, off, m[:3] * 1e-3, rot(np.r_[0.0, m[3:5]])  # mount shift in mm


def residuals(x, views, kin: Kinematics, tags, K) -> np.ndarray:
    R, t, off, mt, mR = unpack(x)
    res = []
    for v in views:
        P = kin.points(v.q, tags[v.tag].corners(), off, mt, mR)
        uv, _ = project(K, R, t, P)
        res.append((uv - v.corners).ravel())
    return np.concatenate(res)


def initial_camera(views, kin: Kinematics, tags, K) -> np.ndarray:
    """The camera from every corner at once, with the nominal kinematics."""
    P = np.vstack([kin.points(v.q, tags[v.tag].corners(), np.zeros(5), np.zeros(3), np.eye(3)) for v in views])
    uv = np.vstack([v.corners for v in views])
    ok, rv, tv = cv2.solvePnP(P, uv, K, None, flags=cv2.SOLVEPNP_SQPNP)
    ok, rv, tv = cv2.solvePnP(P, uv, K, None, rv, tv, True, flags=cv2.SOLVEPNP_ITERATIVE)
    return np.r_[rv.ravel(), tv.ravel()]


def fit(views, kin: Kinematics, tags, K, log=print) -> tuple[np.ndarray, dict]:
    from scipy.optimize import least_squares
    x0 = np.zeros(N_PARAMS)
    x0[:6] = initial_camera(views, kin, tags, K)
    r0 = residuals(x0, views, kin, tags, K)
    log(f"  start: camera from all corners with the nominal arm, rms {np.sqrt(np.mean(r0 ** 2)):.1f} px")
    # Big joint offsets (the elbow's 17 deg) make the first PnP camera poor;
    # a camera-only pass, then everything, then a robust polish.
    s = least_squares(lambda c: residuals(np.r_[c, x0[6:]], views, kin, tags, K), x0[:6], method="lm")
    x0[:6] = s.x
    best = None
    for elbow in (0.0, np.radians(15), -np.radians(15)):  # the elbow offset can be large
        xs = x0.copy()
        xs[6 + FIT_JOINTS.index(2)] = elbow
        s = least_squares(residuals, xs, args=(views, kin, tags, K), method="lm", x_scale="jac")
        if best is None or s.cost < best.cost:
            best = s
    s = least_squares(residuals, best.x, args=(views, kin, tags, K), loss="soft_l1", f_scale=1.0, x_scale="jac")
    # Some detections are wrong by pixels at a corner or two - the back
    # marker's edge run into the black connector and its shadow, seen
    # obliquely. Drop views far worse than the rest, and fit again on those
    # left: the arm's geometry is the same in every view, so a view that
    # disagrees with all the others is the detector's fault.
    first = view_rms(s.fun)
    keep = first <= gate(first)
    dropped = [i for i in range(len(views)) if not keep[i]]
    if dropped:
        kept = [v for v, k in zip(views, keep) if k]
        log(f"  dropped {len(dropped)} of {len(views)} views as detector outliers "
            f"({', '.join(f'{first[i]:.1f}' for i in dropped)} px, against a median of {np.median(first):.2f})")
        s = least_squares(residuals, s.x, args=(kept, kin, tags, K), method="lm", x_scale="jac")
    per = view_rms(s.fun)
    r = s.fun.reshape(-1, 4, 2)
    return s.x, {"rms_px": float(np.sqrt(np.mean(np.sum(r ** 2, -1)))), "per_image_px": per.tolist(),
                 "dropped": len(dropped), "dropped_px": [float(first[i]) for i in dropped]}


def view_rms(fun) -> np.ndarray:
    """RMS corner error of each view (px), from stacked residuals."""
    r = np.asarray(fun).reshape(-1, 4, 2)
    return np.sqrt((r ** 2).sum(-1).mean(-1))


def gate(per) -> float:
    return max(OUTLIER_FLOOR_PX, OUTLIER * float(np.median(per)))


# -- the proof ------------------------------------------------------------------


def run_sim(seed: int, n_fit: int, n_held: int, render: bool, out: Path | None, log=print,
            exact_lens: bool = False) -> dict:
    from lab.experiments import EXPERIMENTS
    from lab.live import Live
    live = Live("handeye", EXPERIMENTS["handeye"]["title"] + f" · seed {seed}", EXPERIMENTS["handeye"]["description"],
                kind="calibration")
    _log = log

    def log(s=""):
        _log(s)
        if s.strip():
            live.log(s)

    rng = np.random.default_rng(seed)
    truth = make_truth(rng)
    K = lens_estimate(truth.K, exact_lens)
    # A lens centre off by d px turns every ray by atan(d / f): the camera's
    # angle can only come out that far off, and it does not matter - the rays
    # through the workspace are what is fitted, and they come out right.
    lens_deg = float(np.degrees(np.arctan(np.hypot(*(K[:2, 2] - truth.K[:2, 2])) / truth.K[0, 0])))
    world = SimWorld(truth, render=render)
    tags = world.tags
    log(f"truth: camera at {np.round(truth.cam_pos * 1000).astype(int)} mm, f {truth.K[0, 0]:.1f} px; "
        f"joint offsets {np.round(np.degrees(truth.offsets), 2)} deg; mount {np.round(truth.mount_t * 1000, 2)} mm, "
        f"{np.degrees(np.linalg.norm(truth.mount_r)):.2f} deg")

    # Plan with a rough guess of the camera: 5 cm and 5 deg off.
    guess_pos = truth.cam_pos + rng.normal(size=3) * 0.05 / np.sqrt(3)
    guess_R = rot(rng.normal(size=3) * np.radians(5) / np.sqrt(3)) @ truth.R
    plan = plan_poses(world, K, guess_R, -guess_R @ guess_pos, n_fit + n_held, rng)
    log(f"planned {len(plan)} poses with the camera guessed "
        f"{np.linalg.norm(guess_pos - truth.cam_pos) * 1000:.0f} mm off")

    views = []
    t0 = time.time()
    for k, (q, want) in enumerate(plan):
        got = world.capture(q, rng)
        if got and got[0][2] is not None:
            img = got[0][2].copy()
            for tid, c, _ in got:
                cv2.polylines(img, [np.int32(np.round(c))], True, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(img, f"id {tid}", tuple(np.int32(c[0]) + [0, -8]), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f"pose {k + 1}/{len(plan)}", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                        cv2.LINE_AA)
            live.frame(img, force=True)
        live.status(pose=f"{k + 1}/{len(plan)}", views=len(views) + len(got))
        for tid, c, img in got:
            views.append(View(q.copy(), tid, c, held=k >= n_fit, image=img if k < 4 or k == n_fit else None))
        if (k + 1) % 10 == 0:
            log(f"  captured {k + 1}/{len(plan)} poses, {len(views)} marker views ({time.time() - t0:.0f} s)")
    fit_views = [v for v in views if not v.held]
    held = [v for v in views if v.held]
    log(f"{len(fit_views)} views to fit, {len(held)} held out; "
        f"{sum(v.tag == 1 for v in views)} of the back marker (id 1)")

    x, st = fit(fit_views, world.kin, tags, K, log=log)
    R, t, off, mt, mR = unpack(x)

    # -- against the truth --
    cam_pos = -R.T @ t
    cam_err_mm = float(np.linalg.norm(cam_pos - truth.cam_pos) * 1000)
    cam_err_deg = angle_deg(R @ truth.R.T)
    off_err = np.degrees(off - truth.offsets)[list(FIT_JOINTS)]
    held_res = residuals(x, held, world.kin, tags, K).reshape(-1, 4, 2) if held else np.zeros((0, 4, 2))
    held_per = np.sqrt((held_res ** 2).sum(-1).mean(-1))
    # The same gate on the held-out views: a detection that bad is rejected
    # in use too. Counted, and the rest scored.
    held_bad = held_per > gate(st["per_image_px"])
    held_ok = held_res[~held_bad]

    def corner_err(vs):
        e = []
        for v in vs:
            loc = tags[v.tag].corners()
            e.append(world.kin.points(v.q, loc, off, mt, mR) - world.true_points(v.q, loc))
        return np.linalg.norm(np.vstack(e), axis=1) if e else np.zeros(0)

    # The paddle face centre, over random poses across the workspace - the
    # point the touch test and the policy actually aim.
    face = np.array([[world.cfg.paddle.blade_centre_x, 0.0, 0.0]])
    fe, tries = [], 0
    arm = world.nominal
    while len(fe) < 300 and tries < 20000:
        tries += 1
        q = rng.uniform(arm.lower, arm.upper)
        if not world.guard.is_safe(q):
            continue
        fe.append(world.kin.points(q, face, off, mt, mR)[0] - world.true_points(q, face)[0])
    fe = np.linalg.norm(np.array(fe), axis=1) * 1000
    # And the same face centre as the camera sees it - camera frame, both
    # errors together: what a ball position in the world is measured against.
    def rms(a):
        return float(np.sqrt(np.mean(np.square(a)))) if len(a) else float("nan")

    ce_held = corner_err(held) * 1000
    ce_fit = corner_err(fit_views) * 1000
    res = {
        "seed": seed, "render": render, "views_fit": len(fit_views), "views_held": len(held),
        "camera_position_err_mm": cam_err_mm, "camera_angle_err_deg": cam_err_deg, "lens_centre_angle_deg": lens_deg,
        "joint_offset_err_deg": dict(zip(["shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                                         map(float, off_err))),
        "joint_offset_fit_deg": dict(zip(["shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                                         map(float, np.degrees(off[list(FIT_JOINTS)])))),
        "joint_offset_true_deg": dict(zip(["shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                                          map(float, np.degrees(truth.offsets[list(FIT_JOINTS)])))),
        "mount_shift_err_mm": float(np.linalg.norm(mt - truth.mount_t) * 1000),
        "mount_tilt_err_deg": angle_deg(mR @ rot(truth.mount_r).T),
        "reproj_rms_px": st["rms_px"], "reproj_per_image_px": st["per_image_px"],
        "views_dropped": st["dropped"], "views_dropped_px": st["dropped_px"],
        "held_reproj_rms_px": rms(np.sqrt((held_ok ** 2).sum(-1)).ravel()) if len(held_ok) else float("nan"),
        "held_rejected": int(held_bad.sum()),
        "held_reproj_per_image_px": held_per.tolist(),
        "corner_rmse_fit_mm": rms(ce_fit), "corner_rmse_held_mm": rms(ce_held),
        "corner_max_held_mm": float(ce_held.max()) if len(ce_held) else float("nan"),
        "face_rmse_mm": rms(fe), "face_max_mm": float(fe.max()), "face_poses": int(len(fe)),
        "camera": {"K": K.tolist(), "R": R.tolist(), "t": t.tolist()},
    }
    res["pass"] = {
        "camera_mm": cam_err_mm <= PASS["cam_mm"], "camera_deg": cam_err_deg <= PASS["cam_deg"] + lens_deg,
        "offsets": bool(np.abs(off_err).max() <= PASS["offset_deg"]),
        "reproj": st["rms_px"] <= PASS["reproj_px"], "face": res["face_max_mm"] <= PASS["face_mm"],
    }
    res["passed"] = all(res["pass"].values())
    live.done(passed=res["passed"], camera_mm=round(cam_err_mm, 3), camera_deg=round(cam_err_deg, 4),
              worst_offset_deg=round(float(np.abs(off_err).max()), 4), fit_px=round(st["rms_px"], 3),
              face_max_mm=round(res["face_max_mm"], 3))

    log("")
    log(f"camera position   {cam_err_mm:6.2f} mm   (pass <= {PASS['cam_mm']})")
    log(f"camera angle      {cam_err_deg:6.3f} deg  (pass <= {PASS['cam_deg']} + {lens_deg:.3f} the lens centre "
        f"error handed in turns every ray by)")
    for j, e in res["joint_offset_err_deg"].items():
        log(f"  {j:14s} true {res['joint_offset_true_deg'][j]:+7.3f}  fit {res['joint_offset_fit_deg'][j]:+7.3f}"
            f"  err {e:+6.3f} deg")
    log(f"paddle mount      {res['mount_shift_err_mm']:.2f} mm, {res['mount_tilt_err_deg']:.3f} deg off")
    log(f"reprojection      fit {st['rms_px']:.3f} px rms ({st['dropped']} views dropped), held-out "
        f"{res['held_reproj_rms_px']:.3f} px rms ({res['held_rejected']} of {len(held)} rejected by the same gate)")
    log(f"marker corners 3D fit {res['corner_rmse_fit_mm']:.2f} mm rms, held-out {res['corner_rmse_held_mm']:.2f} "
        f"mm rms (max {res['corner_max_held_mm']:.2f})")
    log(f"paddle face 3D    {res['face_rmse_mm']:.2f} mm rms, max {res['face_max_mm']:.2f} mm "
        f"over {len(fe)} random poses")
    log("per image (px):  " + " ".join(f"{e:.2f}" for e in st["per_image_px"]))
    log(f"{'PASS' if res['passed'] else 'FAIL'}: " + ", ".join(f"{k} {'ok' if v else 'NO'}" for k, v in res["pass"].items()))

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(res, indent=2) + "\n")
        for k, v in enumerate([v for v in views if v.image is not None][:5]):
            img = v.image.copy()
            P = world.kin.points(v.q, tags[v.tag].corners(), off, mt, mR)
            uv, _ = project(K, R, t, P)
            for a, b in zip(v.corners, uv):
                cv2.circle(img, tuple(np.int32(np.round(a))), 6, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.drawMarker(img, tuple(np.int32(np.round(b))), (255, 0, 255), cv2.MARKER_CROSS, 14, 1)
            cv2.putText(img, f"id {v.tag} {'held out' if v.held else 'fit'}: green = detected, magenta = fit",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(out / f"view{k}.png"), img)
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sim", action="store_true", help="prove it in the sim against a known truth")
    p.add_argument("--no-render", action="store_true", help="exact corners plus 0.15 px noise, no rendering")
    p.add_argument("--exact-lens", action="store_true", help="hand the fit the true lens, not one 0.1%% / 1.5 px off")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--poses", type=int, default=40)
    p.add_argument("--held", type=int, default=12)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    if not a.sim:
        p.error("only --sim exists so far: the real run needs the printed paddle and the lens calibration")
    out = a.out or OUT / time.strftime(f"sim_%Y%m%d_%H%M%S_seed{a.seed}{'_norender' if a.no_render else ''}")
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "w")

    def log(s=""):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    res = run_sim(a.seed, a.poses, a.held, not a.no_render, out, log=log, exact_lens=a.exact_lens)
    log(f"wrote {out}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
