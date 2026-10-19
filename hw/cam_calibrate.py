"""Place the webcam in the arm's world, using the arm itself as the target.

    .venv/bin/python -m hw.cam_calibrate            # move, fit, write hw/camera.json
    .venv/bin/python -m hw.cam_calibrate --refit    # fit again from the last run's points

The arm moves the blue paddle through a set of poses, each checked against the
table in sim. At each one the camera finds the blade and the sim's kinematics
say where the blade's centre is in the world. A dozen or more of those
pairs fix the camera: its focal length, where it is and which way it looks
(a pinhole model; the FaceTime camera's distortion is small enough to leave
out). The blade's apparent size is fitted too, which pins the distance.

Turn Center Stage off first (Control Centre > Video Effects): it crops and
pans the picture to follow faces, which moves the camera under the fit.

Writes hw/camera.json and out/hw/cam_calibration/{points.json, check.jpg}:
check.jpg draws where the fitted camera says the arm is, on the real frame.
"""

from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

from .vision import CAMERA_FILE, PADDLE_R, ROOT, Camera, CameraModel, PaddleFinder

OUT = ROOT / "out" / "hw" / "cam_calibration"

TOWARD_CAMERA = np.array([0.0, -1.0, 0.0])  # the webcam is off the arm's -y side
READY = np.array([0.0, -0.32, 0.87, -0.55, 0.0])  # keep-up start pose, blade turned to the camera
SAMPLE_BOX = (np.array([-0.6, -0.8, 0.2, -1.2, -0.5]), np.array([0.6, 0.4, 1.5, 0.3, 0.5]))


def face_centre(world, q) -> np.ndarray:
    return face(world, q)[0]


def face(world, q):
    import mujoco

    from sim.paddle import face_pose
    world.reset(np.asarray(q, float))
    mujoco.mj_forward(world.model, world.data)
    return face_pose(world.model, world.data)


def blade_centre(world, q, dx: float = 0.0) -> np.ndarray:
    """The blade's centre, moved dx metres further down the roll axis than the
    sim's paddle has it (the axis is the paddle body's x)."""
    import mujoco

    p = face_centre(world, q)
    if dx:
        bid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, "paddle")
        p = p + dx * world.data.xmat[bid].reshape(3, 3)[:, 0]
    return p


def plan_poses(check, n: int = 24, seed: int = 0) -> list[np.ndarray]:
    """Poses that spread the blade's centre through the space the ball will
    use, each clear of the table with the blade turned toward the camera.
    Chosen by farthest-point sampling, so they cover the space evenly."""
    rng = np.random.default_rng(seed)
    qs, ps = [], []
    for q in rng.uniform(*SAMPLE_BOX, size=(3000, 5)):
        if not check.guard.is_safe(q):
            continue
        p, normal = face(check.world, q)
        if normal @ TOWARD_CAMERA < 0.6 or p[2] < 0.08:
            continue
        qs.append(q)
        ps.append(p)
    ps = np.array(ps)
    pick = [int(np.argmin(np.linalg.norm(ps - face_centre(check.world, READY), axis=1)))]
    d = np.linalg.norm(ps - ps[pick[0]], axis=1)
    while len(pick) < min(n, len(qs)):
        k = int(np.argmax(d))
        pick.append(k)
        d = np.minimum(d, np.linalg.norm(ps - ps[k], axis=1))
    # Visit them in a short tour, so consecutive moves are small.
    order, left = [pick[0]], set(pick[1:])
    while left:
        k = min(left, key=lambda j: np.linalg.norm(qs[j] - qs[order[-1]]))
        order.append(k)
        left.remove(k)
    return [qs[k] for k in order]


def collect(arm, cam: Camera, finder: PaddleFinder) -> tuple[list[dict], np.ndarray]:
    """Move through the poses; at each, keep the blue mask and the blade's true centre."""
    shots, frame = [], None
    for k, q in enumerate(plan_poses(arm.check)):
        if not arm.check.path_ok(arm.q(), q):
            arm.move_to(READY, speed=0.5)  # around, rather than through, the table
            if not arm.check.path_ok(arm.q(), q):
                print(f"pose {k}: no clear path in sim, skipped", flush=True)
                continue
        arm.move_to(q, speed=0.5)
        time.sleep(0.5)
        n, masks = 0, []
        for _ in range(3):
            frame, _, n = cam.read(after=n)
            masks.append(finder.mask(frame))
        qm = arm.q()
        shots.append({"q": qm, "p": face_centre(arm.check.world, qm), "mask": np.mean(masks, axis=0) > 0.5})
        print(f"pose {k}: blade centre at {np.round(shots[-1]['p'], 3)} m", flush=True)
    return sightings(shots), frame


def sightings(shots: list[dict]) -> list[dict]:
    """The blade in each shot: blue that is there in this shot but not in most
    of the others. Blue things that stay put - magnets, a shirt - drop out."""
    static = np.mean([s["mask"] for s in shots], axis=0) > 0.5
    pts = []
    for k, s in enumerate(shots):
        m = (s["mask"] & ~static).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(m)
        if n < 2:
            print(f"pose {k}: blade not seen", flush=True)
            continue
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 400:
            print(f"pose {k}: blade too small to trust ({area} px)", flush=True)
            continue
        cnts, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        major = max(cv2.fitEllipse(cnt)[1]) if len(cnt) >= 5 else 2 * np.sqrt(area / np.pi)
        u, v = cents[i]
        pts.append({"q": np.asarray(s["q"]).tolist(), "p": np.asarray(s["p"]).tolist(),
                    "uv": [float(u), float(v)], "major": float(major), "area": area})
        print(f"pose {k}: blade at ({u:.0f}, {v:.0f}) px, {major:.0f} px across", flush=True)
    return pts


JOINT_FIT = (1, 2, 3)  # shoulder lift, elbow, wrist flex: offsets the fit may find
JOINT_PRIOR = 0.1  # rad: how far from the hand calibration they are expected to be


def fit(pts: list[dict], width: int, height: int, world=None, focal: float | None = None,
        joints: bool = True) -> tuple[CameraModel, dict]:
    """Camera (focal length, rotation, position) from blade sightings. Given the
    sim `world`, also small offsets on the lift, elbow and wrist: where the
    blade is depends on the arm's calibration, and fitting the camera alone
    would fold any error there into the camera. (A pan offset cannot be told
    from turning the camera about the base, so pan is left alone.)"""
    P0 = np.array([p["p"] for p in pts])
    Q = np.array([p["q"] for p in pts]) if world is not None and "q" in pts[0] else None
    uv_all = np.array([p["uv"] for p in pts])
    major_all = np.array([p["major"] for p in pts])
    cx, cy = width / 2, height / 2
    nj = len(JOINT_FIT) + 1 if Q is not None and joints else 0  # + the blade's offset down the roll axis
    idx = np.arange(len(pts))

    def model(x):
        f, rv, t = (focal if focal else x[0]), x[1:4], x[4:7]
        return CameraModel(f=f, cx=cx, cy=cy, R=cv2.Rodrigues(rv)[0], t=t, width=width, height=height)

    def points(x, sel):
        if not nj:
            return P0[sel]
        dq = np.zeros(5)
        dq[list(JOINT_FIT)] = x[7:6 + nj]
        return np.array([blade_centre(world, Q[k] + dq, x[6 + nj]) for k in sel])

    def residuals(x, sel):
        m, P = model(x), points(x, sel)
        proj, _ = m.project(P)
        dist = np.linalg.norm(m.to_cam(P), axis=1)
        # The blade's longest image chord is its full diameter whatever its tilt.
        size = 0.3 * (major_all[sel] - 2 * PADDLE_R * m.f / dist)
        prior = np.r_[x[7:6 + nj] / JOINT_PRIOR, x[6 + nj:7 + nj] / 0.03] if nj else np.zeros(0)
        return np.r_[(proj - uv_all[sel]).ravel(), size, 10.0 * np.asarray(prior)]

    def pixel_err(x, sel):
        r = residuals(x, sel)
        return np.hypot(r[0:2 * len(sel):2], r[1:2 * len(sel):2])

    best = None
    for f0 in ([focal] if focal else np.linspace(600, 1500, 10)):
        K = np.array([[f0, 0, cx], [0, f0, cy], [0, 0, 1.0]])
        ok, rv, tv = cv2.solvePnP(P0, uv_all, K, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        r = least_squares(lambda x: residuals(np.r_[x, np.zeros(nj)], idx)[:-nj or None],
                          np.r_[f0, rv.ravel(), tv.ravel()], loss="soft_l1", f_scale=8.0)
        if best is None or r.cost < best.cost:
            best = r
    x = np.r_[best.x, np.zeros(nj)]
    if nj:
        x = least_squares(residuals, x, args=(idx,), loss="soft_l1", f_scale=8.0).x
    # Drop sightings the fit cannot place (a blade half hidden by the arm,
    # something else blue) and fit again without them.
    err = pixel_err(x, idx)
    keep = err < max(15.0, 3 * np.median(err))
    if keep.sum() >= 6 and not keep.all():
        x = least_squares(residuals, x, args=(idx[keep],), loss="soft_l1", f_scale=8.0).x
    m = model(x)
    err = pixel_err(x, idx[keep])
    stats = {"points": int(keep.sum()), "dropped": int((~keep).sum()),
             "rms_px": float(np.sqrt(np.mean(err**2))), "max_px": float(err.max()),
             "hfov_deg": float(np.degrees(2 * np.arctan(width / 2 / m.f))),
             "camera_position_m": m.position.round(3).tolist(),
             "joint_offsets_deg": {["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"][j]:
                                   round(float(np.degrees(v)), 2) for j, v in zip(JOINT_FIT, x[7:6 + nj])},
             "blade_further_out_mm": round(float(x[6 + nj]) * 1000, 1) if nj else 0.0,
             "residual_px": err.round(1).tolist()}
    m.joint_offsets = np.zeros(5)
    if nj:
        m.joint_offsets[list(JOINT_FIT)] = x[7:6 + nj]
    m.blade_dx = float(x[6 + nj]) if nj else 0.0
    return m, stats


def check_image(frame: np.ndarray, m: CameraModel, pts: list[dict], world=None) -> np.ndarray:
    img = frame.copy()
    for p in pts:
        P = blade_centre(world, np.array(p["q"]) + m.joint_offsets, m.blade_dx) if world is not None else p["p"]
        (u, v), = m.project(np.array([P]))[0]
        cv2.circle(img, (int(p["uv"][0]), int(p["uv"][1])), 6, (0, 255, 0), 2)
        cv2.drawMarker(img, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
    # The table edge under the arm and the world axes, from the origin (arm base).
    axes = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    a, z = m.project(axes)
    if np.all(z > 0):
        for k, c in zip((1, 2, 3), ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
            cv2.line(img, tuple(a[0].astype(int)), tuple(a[k].astype(int)), c, 3)
    roi = m.roi()
    if roi:
        cv2.rectangle(img, roi[:2], roi[2:], (255, 255, 0), 2)
    cv2.putText(img, "green: blade seen   red: fitted camera's projection   axes x red, y green, z blue at the arm base",
                (14, img.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def apply_offsets(dq: np.ndarray) -> None:
    """The arm reads q but stands at q + dq in the sim's terms: shift each joint's
    centre so it reads q + dq, and the fold path with it."""
    from .so101 import CALIBRATION, RAD_PER_TICK
    cal = json.loads(CALIBRATION.read_text())
    for j, name in enumerate(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]):
        cal[name]["center"] = round(cal[name]["center"] - cal[name]["sign"] * dq[j] / RAD_PER_TICK, 1)
    cal["_fold"] = [[round(v + d, 4) for v, d in zip(w, dq)] for w in cal["_fold"]]
    cal["_camera_fit_offsets_deg"] = np.degrees(dq).round(2).tolist()
    CALIBRATION.write_text(json.dumps(cal, indent=2))
    print("arm calibration shifted by", np.degrees(dq).round(2), "deg", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refit", action="store_true")
    p.add_argument("--focal", type=float, help="hold the focal length (pixels) instead of fitting it")
    p.add_argument("--no-joints", action="store_true", help="trust the arm calibration: fit the camera only")
    p.add_argument("--apply-offsets", action="store_true",
                   help="move the fitted joint offsets into hw/calibration.json, so the arm reports sim angles")
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.refit:
        saved = json.loads((OUT / "points.json").read_text())
        pts, size = saved["points"], saved["size"]
        frame = cv2.imread(str(OUT / "last_frame.jpg"))
    else:
        from .so101 import SO101
        cam = Camera()
        arm = SO101()
        arm.configure(acceleration=30, torque_limit=600)
        arm.torque(True)
        try:
            arm.unfold()
            pts, frame = collect(arm, cam, PaddleFinder())
        finally:
            arm.hold()
            arm.fold()
            arm.torque(False)
            cam.close()
        size = [frame.shape[1], frame.shape[0]]
        (OUT / "points.json").write_text(json.dumps({"points": pts, "size": size}, indent=1))
        cv2.imwrite(str(OUT / "last_frame.jpg"), frame)

    if len(pts) < 6:
        raise SystemExit(f"only {len(pts)} blade sightings; need 6 or more")
    from .so101 import SimCheck
    m, stats = fit(pts, *size, world=SimCheck().world, focal=args.focal, joints=not args.no_joints)
    print(json.dumps(stats, indent=1), flush=True)
    if args.apply_offsets:
        apply_offsets(m.joint_offsets)
        stats["applied_to_arm_calibration"] = True
        m.joint_offsets = np.zeros(5)
    m.save(CAMERA_FILE, fit=stats)
    cv2.imwrite(str(OUT / "check.jpg"), check_image(frame, m, pts, SimCheck().world))
    print("wrote", CAMERA_FILE, "and", OUT / "check.jpg", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
