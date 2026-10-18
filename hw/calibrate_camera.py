"""Calibrate a camera's lens from a printed checkerboard.

    .venv/bin/python -m hw.calibrate_camera board                       # write a printable board
    .venv/bin/python -m hw.calibrate_camera capture --camera facetime   # hold the board up; ~2 minutes
    .venv/bin/python -m hw.calibrate_camera solve --camera facetime --dir out/hw/calib_facetime_<stamp>

1. `board` writes out/hw/checkerboard.png: 9 x 6 inner corners (10 x 7
   squares of 25 mm) on a landscape Letter/A4 page at 300 dpi. Print it at 100% scale,
   tape it FLAT to something rigid, and measure one square with a ruler - pass
   that measurement as --square-mm, since printers scale.
2. `capture` watches the camera and saves a view each time the board is held
   still somewhere new: near each edge and corner of the image, close and far,
   and tilted different ways. Progress and a preview (out/hw/calib_preview.jpg,
   or a window with --show) say what is still missing. It solves at the end.
3. `solve` runs OpenCV's calibration on saved views: focal lengths, optical
   centre and lens distortion, with the reprojection error in pixels as the
   quality check (under ~0.5 px is good). Views far worse than the rest are
   dropped once and the solve repeated. The result goes into
   hw/cameras/<camera>.json, where everything downstream reads it.

The FaceTime camera has fixed focus, so its calibration does not drift; run it
at the mode the tracker uses (1280x720) - other modes have other intrinsics.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PREVIEW = ROOT / "out" / "hw" / "calib_preview.jpg"


# -- the board -----------------------------------------------------------------


def make_board(path: Path, cols: int = 9, rows: int = 6, square_mm: float = 25.0, dpi: int = 300) -> Path:
    """A checkerboard with cols x rows INNER corners, centred on a landscape
    Letter page (it also fits landscape A4), at `dpi`."""
    import cv2

    px = square_mm / 25.4 * dpi
    w, h = int(11 * dpi), int(8.5 * dpi)  # landscape: 10 squares of 25 mm are 250 mm wide
    bw, bh = int(round((cols + 1) * px)), int(round((rows + 1) * px))
    if bw > w - dpi // 2 or bh > h - dpi // 2:
        raise ValueError(f"a {cols + 1}x{rows + 1} board of {square_mm} mm squares does not fit the page")
    img = np.full((h, w), 255, np.uint8)
    x0, y0 = (w - bw) // 2, (h - bh) // 2
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                img[y0 + int(round(r * px)): y0 + int(round((r + 1) * px)),
                    x0 + int(round(c * px)): x0 + int(round((c + 1) * px))] = 0
    cv2.putText(img, f"{cols}x{rows} inner corners, {square_mm:g} mm squares - print at 100%, measure, tape flat",
                (x0, y0 + bh + int(0.4 * dpi)), cv2.FONT_HERSHEY_SIMPLEX, dpi / 300, 0, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path


def board_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    """The inner corners in the board's own plane, in metres."""
    g = np.zeros((rows * cols, 3), np.float32)
    g[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm / 1000.0
    return g


def find_corners(gray, cols: int, rows: int):
    """Sub-pixel inner corners, or None."""
    import cv2

    ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ok:
        return corners.astype(np.float32)
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows), cv2.CALIB_CB_ADAPTIVE_THRESH)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    return cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), crit)


# -- solving -----------------------------------------------------------------------


def solve(corner_sets: list[np.ndarray], image_size: tuple[int, int], cols: int, rows: int,
          square_mm: float) -> dict:
    """Intrinsics and distortion from views of the board. image_size is (w, h)."""
    import cv2

    obj = board_points(cols, rows, square_mm)

    def run(sets):
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera([obj] * len(sets), sets, image_size, None, None)
        per = []
        for c, r, t in zip(sets, rvecs, tvecs):
            proj, _ = cv2.projectPoints(obj, r, t, K, dist)
            per.append(float(np.sqrt(np.mean(np.sum((proj - c) ** 2, axis=2)))))
        return rms, K, dist, np.array(per)

    rms, K, dist, per = run(corner_sets)
    dropped = []
    if len(corner_sets) > 8:
        keep = per < max(2.0 * np.median(per), 0.3)
        if not keep.all():
            dropped = [int(i) for i in np.flatnonzero(~keep)]
            rms, K, dist, per = run([c for c, k in zip(corner_sets, keep) if k])
    w, h = image_size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return {
        "K": np.round(K, 4).tolist(), "dist": np.round(dist.ravel(), 6).tolist(),
        "rms_px": round(float(rms), 4), "per_view_px": np.round(per, 3).tolist(), "dropped_views": dropped,
        "image_size": [w, h], "n_views": len(corner_sets) - len(dropped),
        "hfov_deg": round(float(np.degrees(2 * np.arctan(w / (2 * fx)))), 2),
        "vfov_deg": round(float(np.degrees(2 * np.arctan(h / (2 * fy)))), 2),
        "centre_offset_px": [round(float(cx - w / 2), 1), round(float(cy - h / 2), 1)],
        "board": {"cols": cols, "rows": rows, "square_mm": square_mm},
        "date": time.strftime("%Y-%m-%d"),
    }


def describe(cal: dict) -> str:
    K = np.array(cal["K"])
    q = "good" if cal["rms_px"] < 0.5 else "usable" if cal["rms_px"] < 1.0 else "poor - recapture"
    return (f"  reprojection error {cal['rms_px']:.3f} px ({q}) over {cal['n_views']} views"
            + (f", {len(cal['dropped_views'])} dropped" if cal["dropped_views"] else "") + "\n"
            f"  focal length fx {K[0, 0]:.1f} px, fy {K[1, 1]:.1f} px\n"
            f"  field of view {cal['hfov_deg']:.1f} x {cal['vfov_deg']:.1f} deg\n"
            f"  optical centre {cal['centre_offset_px'][0]:+.1f}, {cal['centre_offset_px'][1]:+.1f} px off the image centre\n"
            f"  distortion k1 {cal['dist'][0]:+.4f}, k2 {cal['dist'][1]:+.4f}")


def solve_dir(directory: Path, cols: int, rows: int, square_mm: float) -> tuple[dict, tuple[int, int]]:
    import cv2

    sets, size = [], None
    for p in sorted(directory.glob("*.png")):
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        size = (gray.shape[1], gray.shape[0])
        c = find_corners(gray, cols, rows)
        if c is not None:
            sets.append(c)
    if len(sets) < 6:
        raise SystemExit(f"only {len(sets)} usable views in {directory}; need at least 6, ideally 20+")
    return solve(sets, size, cols, rows, square_mm), size


# -- capturing -------------------------------------------------------------------


def coverage(sets: list[np.ndarray], size: tuple[int, int]) -> tuple[np.ndarray, list[str]]:
    """Which ninths of the image the saved views' boards have covered, and what
    to do next."""
    w, h = size
    grid = np.zeros((3, 3), int)
    areas = []
    for c in sets:
        pts = c.reshape(-1, 2)
        cx, cy = pts.mean(axis=0)
        grid[min(2, int(3 * cy / h)), min(2, int(3 * cx / w))] += 1
        areas.append(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1]) / (w * h))
    tips = []
    names = [["top-left", "top", "top-right"], ["left", "centre", "right"], ["bottom-left", "bottom", "bottom-right"]]
    missing = [names[r][c] for r in range(3) for c in range(3) if grid[r, c] == 0]
    if missing:
        tips.append("move the board to: " + ", ".join(missing))
    if areas and max(areas) < 0.25:
        tips.append("bring it closer, filling about half the image")
    if areas and min(areas) > 0.08:
        tips.append("hold it further away too")
    if len(sets) >= 6:
        tips.append("tilt it left/right and up/down by ~30 deg for a few views")
    return grid, tips


def is_new(c: np.ndarray, sets: list[np.ndarray], size: tuple[int, int]) -> bool:
    """Far enough from every saved view, in position or apparent size, to add information."""
    diag = np.hypot(*size)
    for s in sets:
        moved = np.mean(np.linalg.norm(c.reshape(-1, 2) - s.reshape(-1, 2), axis=1)) / diag
        if moved < 0.06:
            return False
    return True


def capture(camera: str, cols: int, rows: int, square_mm: float, count: int, show: bool) -> Path:
    import cv2

    from hw.camera import open_camera

    out = ROOT / "out" / "hw" / f"calib_{camera}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    sets, prev, last_preview, last_tip = [], None, 0.0, 0.0
    print(f"Hold the {cols}x{rows} board up to the camera. Saving a view whenever it is still "
          f"somewhere new; {count} wanted. Preview: {PREVIEW}" + (" (and a window)" if show else ""))
    with open_camera(camera) as cam:
        size = (cam.spec.width, cam.spec.height)
        while len(sets) < count:
            frame, _ = cam.wait_new()
            if frame is None:
                break
            size = (frame.shape[1], frame.shape[0])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            c = find_corners(gray, cols, rows)
            still = (c is not None and prev is not None
                     and np.mean(np.linalg.norm(c.reshape(-1, 2) - prev.reshape(-1, 2), axis=1)) < 1.5)
            if still and is_new(c, sets, size):
                cv2.imwrite(str(out / f"{len(sets):02d}.png"), frame)
                sets.append(c)
                grid, tips = coverage(sets, size)
                print(f"  saved {len(sets)}/{count}" + (f"  - next: {tips[0]}" if tips else ""), flush=True)
            prev = c
            now = time.monotonic()
            if now - last_preview > 0.3 or show:
                vis = frame.copy()
                if c is not None:
                    cv2.drawChessboardCorners(vis, (cols, rows), c, True)
                for s in sets:  # where saved views were
                    cv2.circle(vis, tuple(int(v) for v in s.reshape(-1, 2).mean(axis=0)), 6, (0, 200, 0), -1)
                cv2.putText(vis, f"{len(sets)}/{count} saved" + ("" if c is not None else "  - board not seen"),
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imwrite(str(PREVIEW), vis)
                last_preview = now
                if show:
                    cv2.imshow("calibration", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            if now - last_tip > 8 and c is None:
                print("  (board not seen - hold it flat, fully in view, well lit)", flush=True)
                last_tip = now
    if show:
        cv2.destroyAllWindows()
    return out


def write_calibration(camera: str, cal: dict) -> Path:
    from hw.camera import CameraSpec

    spec = CameraSpec.load(camera)
    if [spec.width, spec.height] != cal["image_size"]:
        print(f"note: calibrated at {cal['image_size']}, spec says {spec.width}x{spec.height}; spec updated")
        spec.width, spec.height = cal["image_size"]
    spec.calibration = cal
    return spec.save()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["board", "capture", "solve"])
    p.add_argument("--camera", default="facetime")
    p.add_argument("--cols", type=int, default=9, help="inner corners across")
    p.add_argument("--rows", type=int, default=6, help="inner corners down")
    p.add_argument("--square-mm", type=float, default=25.0, help="measured size of one printed square")
    p.add_argument("--count", type=int, default=25)
    p.add_argument("--dir", type=Path, default=None, help="saved views to solve from")
    p.add_argument("--show", action="store_true", help="also show a live window")
    p.add_argument("--out", type=Path, default=ROOT / "out" / "hw" / "checkerboard.png")
    args = p.parse_args(argv)

    if args.command == "board":
        print("wrote", make_board(args.out, args.cols, args.rows, args.square_mm))
        print("Print landscape at 100% scale (no 'fit to page'), measure one square, tape it flat to card or board.")
        return 0
    directory = args.dir
    if args.command == "capture":
        directory = capture(args.camera, args.cols, args.rows, args.square_mm, args.count, args.show)
    if directory is None:
        raise SystemExit("solve needs --dir")
    cal, _ = solve_dir(directory, args.cols, args.rows, args.square_mm)
    print(f"{args.camera} calibration:\n{describe(cal)}")
    print("wrote", write_calibration(args.camera, cal))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
