"""Prove the lens calibration in sim, where the true lens is known.

    .venv/bin/python -m hw.sim_intrinsics                 # 80 views + 10 held out, seed 0
    .venv/bin/python -m hw.sim_intrinsics --views 60 --seed 3

The printed ChArUco board (cad/charuco.py) is rendered in MuJoCo at random
poses - across the whole frame, 0.3-0.8 m away, tilted up to 60 degrees - by a
camera whose intrinsics are set exactly (focal and principal point in pixels).
MuJoCo renders a perfect pinhole, so each frame is rendered wider than the
camera and then warped through a known lens distortion: OpenCV's full
14-coefficient model (radial k1-k6 in rational form, tangential p1 p2, thin
prism s1-s4, sensor tilt tauX tauY), set to values typical of a small webcam
lens. Then the frames go through the same calibration the real camera will:
ChArUco detection, cv2.calibrateCameraExtended with the 14-coefficient model.

Reported against the truth:
  reprojection   RMS over the calibration views, and on held-out views
  intrinsics     fx, fy, cx, cy errors
  distortion     how far the fitted lens and the true lens disagree about
                 where each pixel's ray goes, over the whole image, in pixels
                 (comparing coefficients one by one means little: different
                 sets can describe nearly the same lens)
  3D             each held-out board, located with the fitted lens: corner
                 position error against the true pose, in millimetres

Writes out/hw/sim_intrinsics/<stamp>/{report.txt, views.jpg}.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cad"))
import charuco  # noqa: E402

W, H = 1280, 720
# The true lens. Focal and centre near a 720p webcam's; distortion typical of a
# small wide lens (mostly barrel), with small decentring, prism and tilt terms.
TRUE_K = np.array([[905.0, 0.0, 651.5], [0.0, 908.0, 352.8], [0.0, 0.0, 1.0]])
TRUE_DIST = np.array([-0.32, 0.14, 0.0009, -0.0006, -0.03,  # k1 k2 p1 p2 k3
                      0.02, -0.01, 0.006,  # k4 k5 k6
                      0.0008, -0.0003, 0.0005, -0.0002,  # s1 s2 s3 s4
                      0.0015, -0.0010])  # tauX tauY (rad)
# How the board is posed. Measured in sim: 0.35-1.2 m and 50 deg left the lens
# centre 70 px out; 0.3-0.8 m and 60 deg brought the fit to 0.06 px and the
# lens to within 1-2 px. Closer, steeper views are what pin a lens down.
DISTANCE = (0.30, 0.80)
MAX_TILT = 60.0
MIN_CORNERS = 8  # fewer, nearly edge-on, and the plane fit for that view degenerates
FLAGS = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL | cv2.CALIB_TILTED_MODEL
TEX_PX_PER_MM = 8.0


def scene(pad_w: int, pad_h: int, cx: float, cy: float) -> mujoco.MjModel:
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth, spec.visual.global_.offheight = pad_w, pad_h
    spec.visual.quality.offsamples = 8
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    img = charuco.raster(TEX_PX_PER_MM)
    tex = spec.add_texture()
    tex.name, tex.type = "board", mujoco.mjtTexture.mjTEXTURE_2D
    rgb = np.repeat(img[:, :, None], 3, axis=2)
    tex.height, tex.width, tex.nchannel = rgb.shape[0], rgb.shape[1], 3
    tex.data = rgb.tobytes()
    mat = spec.add_material()
    mat.name = "board"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "board"
    mat.texrepeat, mat.texuniform = [1.0, 1.0], False
    mat.specular = mat.shininess = mat.reflectance = 0.0
    body = spec.worldbody.add_body(name="board", mocap=True)
    g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[charuco.WIDTH / 2000, charuco.HEIGHT / 2000, charuco.PLATE_THICK / 2000])
    g.material = "board"
    # A grey wall far behind, so the board is not on black.
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, -4.0], size=[20, 20, 0.01],
                            rgba=[0.55, 0.55, 0.52, 1])
    cam = spec.worldbody.add_camera(name="cam", pos=[0, 0, 0])
    # MuJoCo cameras look down their -z with +y up; ours looks down +z with y down.
    cam.quat = [0.0, 1.0, 0.0, 0.0]
    cam.resolution = [pad_w, pad_h]
    cam.focal_pixel = [TRUE_K[0, 0], TRUE_K[1, 1]]
    # principal_pixel is an offset from the image centre, which MuJoCo applies
    # with the opposite sign to OpenCV's (checked: tests/test_sim_intrinsics.py).
    # And MuJoCo's pixel centres sit at +0.5 where OpenCV's are whole numbers.
    cam.principal_pixel = [pad_w / 2 - (cx + 0.5), pad_h / 2 - (cy + 0.5)]
    cam.sensor_size = [pad_w * 1e-5, pad_h * 1e-5]
    return spec.compile()


def board_corners_3d() -> np.ndarray:
    """ChArUco corners in the board body's frame (m): the texture spans the box
    face, x right and y up as rendered on its +z face."""
    b, _ = charuco.board()
    c = b.getChessboardCorners()[:, :2] + charuco.FRAME  # mm, x right, y down from the top-left
    x = c[:, 0] / 1000 - charuco.WIDTH / 2000
    y = charuco.HEIGHT / 2000 - c[:, 1] / 1000
    return np.c_[x, y, np.full(len(c), charuco.PLATE_THICK / 2000)]


def random_pose(rng) -> tuple[np.ndarray, np.ndarray]:
    """A board pose in the camera frame (x right, y down, z forward): somewhere
    in the frame, at some distance, tilted, rolled, facing the camera."""
    z = rng.uniform(*DISTANCE)
    u, v = rng.uniform(0.05, 0.95) * W, rng.uniform(0.05, 0.95) * H
    xn, yn = (u - TRUE_K[0, 2]) / TRUE_K[0, 0], (v - TRUE_K[1, 2]) / TRUE_K[1, 1]
    t = np.array([xn * z, yn * z, z])
    # Board +z must face the camera: start from board z = -camera z, then tilt and roll.
    R0 = np.diag([1.0, -1.0, -1.0])
    axis = rng.normal(size=3)
    axis[2] = 0
    axis /= np.linalg.norm(axis)
    tilt = cv2.Rodrigues(axis * np.radians(rng.uniform(0, MAX_TILT)))[0]
    roll = cv2.Rodrigues(np.array([0, 0, rng.uniform(-np.pi, np.pi)]))[0]
    return tilt @ R0 @ roll, t


def to_world(R_cb, t_cb):
    """Our camera sits at the world origin, looking down +z of the world's
    camera-convention frame: the camera quat above makes camera-frame == world
    frame up to the MuJoCo axis flip, so world = camera here."""
    return R_cb, t_cb


def render_pinhole(model, data, r, R_cb, t_cb) -> np.ndarray:
    data.mocap_pos[0] = t_cb
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R_cb.ravel())
    data.mocap_quat[0] = q
    mujoco.mj_forward(model, data)
    r.update_scene(data, camera="cam")
    return cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)


def undistorted_grid() -> np.ndarray:
    """For each pixel of the distorted (real) image, its ray as normalised
    pinhole coordinates (x/z, y/z), shape (H, W, 2)."""
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    pts = np.stack([uu.ravel(), vv.ravel()], axis=1)[:, None, :]
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)
    return cv2.undistortPoints(pts, TRUE_K, TRUE_DIST, None, None, criteria=crit)[:, 0, :].reshape(H, W, 2)


def render_size(und: np.ndarray, pad: int = 12) -> tuple[int, int, float, float]:
    """The pinhole render that covers every ray the distorted image looks
    along: width, height, and its principal point."""
    xs, ys = und[..., 0] * TRUE_K[0, 0], und[..., 1] * TRUE_K[1, 1]
    left, right = -xs.min() + pad, xs.max() + pad
    top, bottom = -ys.min() + pad, ys.max() + pad
    w, h = int(np.ceil(left + right)), int(np.ceil(top + bottom))
    w += w % 2
    h += h % 2
    return w, h, left, top


def detect(img, board, detector):
    corners, ids, _, _ = detector.detectBoard(img)
    if ids is None or len(ids) < MIN_CORNERS:
        return None
    obj, imgp = board.matchImagePoints(corners, ids)
    return obj.astype(np.float32), imgp.astype(np.float32), ids.ravel()


STAGES = (  # each fit starts from the last: a 14-term fit from nothing finds wrong minima
    ("5 (k1 k2 p1 p2 k3)", 0, 5),
    ("8 (+ rational k4-k6)", cv2.CALIB_RATIONAL_MODEL, 8),
    # The image centre is held from here on. Measured in sim: with it free, the
    # prism and tilt terms trade against it - even a fit started at the true
    # lens drifts 25 px off-centre at the same 0.064 px RMS - while the 5- and
    # 8-term fits put it within 1-2 px. So the richer terms refine the shape of
    # the distortion around a centre the simpler model has already found.
    ("12 (+ thin prism s1-s4)", cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL
     | cv2.CALIB_FIX_PRINCIPAL_POINT, 12),
    ("14 (+ tilt tauX tauY)", FLAGS | cv2.CALIB_FIX_PRINCIPAL_POINT, 14),
)


def calibrate(obj, img, size=(W, H)):
    """Staged: the 5-term model from a rough guess, then each richer model
    starting from the one before. Returns the 14-term result and every stage's RMS."""
    K = np.array([[1000.0, 0, size[0] / 2], [0, 1000.0, size[1] / 2], [0, 0, 1]])
    dist = np.zeros(14)
    stages = []
    for name, flags, n in STAGES:
        guess = cv2.CALIB_USE_INTRINSIC_GUESS if stages else 0
        rms, K, d, rvecs, tvecs, std_in, _, _ = cv2.calibrateCameraExtended(obj, img, size, K, dist[:n].copy(),
                                                                             flags=flags | guess)
        dist = np.zeros(14)
        dist[:d.size] = d.ravel()
        stages.append((name, float(rms)))
    return rms, K, dist, rvecs, tvecs, std_in, stages


def ray_error(K, dist, within=None) -> tuple[float, float]:
    """Over the image, how far apart (pixels) the fitted and the true lens put
    each pixel's ray, measured back through the true pinhole. `within`: a
    convex hull (N, 2) to restrict it to - the area the views covered."""
    uu, vv = np.meshgrid(np.linspace(0, W - 1, 64), np.linspace(0, H - 1, 36))
    pts = np.stack([uu.ravel(), vv.ravel()], axis=1)
    if within is not None:
        pts = pts[[cv2.pointPolygonTest(within, (float(x), float(y)), False) >= 0 for x, y in pts]]
    pts = pts[:, None, :]
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
    a = cv2.undistortPoints(pts, TRUE_K, TRUE_DIST, None, TRUE_K, criteria=crit)[:, 0]
    b = cv2.undistortPoints(pts, K, dist, None, TRUE_K, criteria=crit)[:, 0]
    e = np.linalg.norm(a - b, axis=1)
    return float(np.sqrt((e**2).mean())), float(e.max())


def run(n_views: int, n_hold: int, seed: int, out: Path | None = None) -> dict:
    rng = np.random.default_rng(seed)
    und = undistorted_grid()
    pad_w, pad_h, cx_r, cy_r = render_size(und)
    model = scene(pad_w, pad_h, cx_r, cy_r)
    data = mujoco.MjData(model)
    r = mujoco.Renderer(model, pad_h, pad_w)
    mx = (und[..., 0] * TRUE_K[0, 0] + cx_r).astype(np.float32)
    my = (und[..., 1] * TRUE_K[1, 1] + cy_r).astype(np.float32)
    board, _ = charuco.board()
    detector = cv2.aruco.CharucoDetector(board)
    X = board_corners_3d()

    views, tiles = [], []
    tries = 0
    while len(views) < n_views + n_hold and tries < 20 * (n_views + n_hold):
        tries += 1
        R, t = random_pose(rng)
        img = cv2.remap(render_pinhole(model, data, r, R, t), mx, my, cv2.INTER_LINEAR)
        d = detect(img, board, detector)
        if d is None:
            continue
        views.append({"R": R, "t": t, "obj": d[0], "img": d[1], "ids": d[2]})
        if len(tiles) < 24:
            t_img = img.copy()
            for p in d[1].reshape(-1, 2):
                cv2.circle(t_img, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)
            tiles.append(cv2.resize(t_img, (320, 180)))
    cal, hold = views[:n_views], views[n_views:]

    rms, K, dist, rvecs, tvecs, std_in, stages = calibrate([v["obj"] for v in cal], [v["img"] for v in cal])

    # Held out: locate each board with the fitted lens, then compare to the truth.
    held_rms, mm = [], []
    for v in hold:
        ok, rv, tv = cv2.solvePnP(v["obj"], v["img"], K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        proj, _ = cv2.projectPoints(v["obj"], rv, tv, K, dist)
        held_rms.append(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - v["img"].reshape(-1, 2)) ** 2, axis=1))))
        Rf = cv2.Rodrigues(rv)[0]
        # The fitted pose maps OpenCV's board frame (mm, from the top-left corner).
        corners_fit = ((Rf @ v["obj"].reshape(-1, 3).T).T + tv.ravel()) / 1000
        corners_true = (v["R"] @ X[v["ids"]].T).T + v["t"]
        mm.append(1000 * np.linalg.norm(corners_fit - corners_true, axis=1))
    mm = np.concatenate(mm) if mm else np.zeros(1)
    # The calibration views' own poses too.
    mm_cal = []
    for v, rv, tv in zip(cal, rvecs, tvecs):
        Rf = cv2.Rodrigues(rv)[0]
        mm_cal.append(1000 * np.linalg.norm(((Rf @ v["obj"].reshape(-1, 3).T).T + tv.ravel()) / 1000
                                            - ((v["R"] @ X[v["ids"]].T).T + v["t"]),
                                            axis=1))
    mm_cal = np.concatenate(mm_cal)
    ray_rms, ray_max = ray_error(K, dist)
    hull = cv2.convexHull(np.concatenate([v["img"].reshape(-1, 2) for v in cal]).astype(np.float32))
    cov_rms, cov_max = ray_error(K, dist, hull)
    covered = cv2.contourArea(hull) / (W * H)

    res = {
        "views": len(cal), "held_out": len(hold), "corners": int(sum(len(v["ids"]) for v in cal)),
        "rms_px": float(rms), "held_rms_px": float(np.sqrt(np.mean(np.square(held_rms)))) if held_rms else np.nan,
        "fx_err_pct": float(100 * (K[0, 0] / TRUE_K[0, 0] - 1)), "fy_err_pct": float(100 * (K[1, 1] / TRUE_K[1, 1] - 1)),
        "cx_err_px": float(K[0, 2] - TRUE_K[0, 2]), "cy_err_px": float(K[1, 2] - TRUE_K[1, 2]),
        "fx_std_pct": float(100 * std_in[0, 0] / K[0, 0]),
        "lens_ray_rms_px": ray_rms, "lens_ray_max_px": ray_max,
        "covered_ray_rms_px": cov_rms, "covered_ray_max_px": cov_max, "covered_frac": covered,
        "held_3d_rmse_mm": float(np.sqrt(np.mean(mm**2))), "held_3d_max_mm": float(mm.max()),
        "cal_3d_rmse_mm": float(np.sqrt(np.mean(mm_cal**2))),
        "K": K.round(2).tolist(), "dist": dist.round(5).tolist(), "stages": stages,
    }
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(out / "views.jpg"), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    return res


def report(res: dict) -> str:
    return "\n".join(["stages        " + "   ".join(f"{n}: {r:.3f} px" for n, r in res["stages"]),
        f"{res['views']} calibration views ({res['corners']} corners), {res['held_out']} held out; "
        "14-coefficient model (rational + thin prism + tilted)",
        f"reprojection  RMS {res['rms_px']:.3f} px on the calibration views, {res['held_rms_px']:.3f} px held out",
        f"intrinsics    fx {res['fx_err_pct']:+.3f}%  fy {res['fy_err_pct']:+.3f}%  "
        f"cx {res['cx_err_px']:+.2f} px  cy {res['cy_err_px']:+.2f} px   (fx 1-sigma {res['fx_std_pct']:.3f}%)",
        f"lens          fitted vs true ray per pixel, where the views reached ({res['covered_frac']:.0%} of the "
        f"image): RMS {res['covered_ray_rms_px']:.2f} px, worst {res['covered_ray_max_px']:.2f} px; whole image "
        f"RMS {res['lens_ray_rms_px']:.2f}, worst {res['lens_ray_max_px']:.2f} px",
        f"3D            held-out board corners vs truth: RMSE {res['held_3d_rmse_mm']:.3f} mm, worst "
        f"{res['held_3d_max_mm']:.3f} mm  (calibration views {res['cal_3d_rmse_mm']:.3f} mm)",
    ])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--views", type=int, default=80)
    p.add_argument("--hold", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    out = ROOT / "out" / "hw" / "sim_intrinsics" / time.strftime("%Y%m%d_%H%M%S")
    res = run(args.views, args.hold, args.seed, out)
    text = report(res)
    print(text, flush=True)
    (out / "report.txt").write_text(text + "\n\nK " + str(res["K"]) + "\ndist " + str(res["dist"]) + "\n")
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
