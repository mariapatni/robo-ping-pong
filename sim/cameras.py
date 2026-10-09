"""Fixed cameras: where they go, what they can see, and how well two of them
can triangulate a ball.

The plan this module serves: no wrist cameras. One fixed USB camera per side,
mounted behind and above its own arm and angled down, with the two fused by a
single Kalman filter into one ball estimate in the table frame. Everything here
is either (a) the pose arithmetic that puts such a camera into the scene, or
(b) a measurement that says whether a given pose is any good - full table in
frame, ball occluded by the arm, pixels across the ball, and how badly
conditioned the two-camera triangulation is.

The projection is a plain pinhole built from the SAME numbers handed to MuJoCo
(sensor size, focal length, resolution), so what `project` says and what the
renderer draws agree. That matters: the whole point of the study is that the
montage and the numbers describe one camera, not two different ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import CameraConfig, CameraRig, TableConfig

CAM_A = "cam_a"
CAM_B = "cam_b"
WORLD_UP = np.array([0.0, 0.0, 1.0])


# -- pose ---------------------------------------------------------------------


def _basis(forward: np.ndarray) -> np.ndarray:
    """Camera-to-world rotation for a camera looking along `forward`.

    MuJoCo's camera looks down its own -z with +y up in the image, so z_cam is
    -forward and y_cam is world up with the along-axis component removed. Columns
    are (right, up, back).
    """
    z = -np.asarray(forward, dtype=float)
    z = z / np.linalg.norm(z)
    y = WORLD_UP - np.dot(WORLD_UP, z) * z
    n = np.linalg.norm(y)
    if n < 1e-9:  # straight down: pick the +x image-up convention
        y = np.array([1.0, 0.0, 0.0]) - z[0] * z
        n = np.linalg.norm(y)
    y = y / n
    x = np.cross(y, z)
    return np.column_stack((x, y, z))


@dataclass(frozen=True)
class Pinhole:
    """A camera placed in the world, with the intrinsics we intend to buy."""

    pos: np.ndarray
    rot: np.ndarray  # camera-to-world, columns (right, up, back)
    cfg: CameraConfig
    name: str = ""

    @property
    def forward(self) -> np.ndarray:
        return -self.rot[:, 2]

    def to_camera(self, p: np.ndarray) -> np.ndarray:
        return self.rot.T @ (np.asarray(p, dtype=float) - self.pos)

    def project(self, p: np.ndarray) -> tuple[float, float, float]:
        """World point -> (u, v, depth) in pixels, v measured downwards.

        `depth` is along the optical axis and is negative for a point behind the
        camera, which is how `in_frame` rejects it.
        """
        w, h = self.cfg.resolution
        fx, fy = self.cfg.focal_px
        c = self.to_camera(p)
        depth = -c[2]
        if abs(depth) < 1e-9:
            return (float("nan"), float("nan"), float(depth))
        return (0.5 * w + fx * c[0] / depth, 0.5 * h - fy * c[1] / depth, float(depth))

    def in_frame(self, p: np.ndarray, margin_px: float = 0.0) -> bool:
        w, h = self.cfg.resolution
        u, v, d = self.project(p)
        if not (d > 0.0) or not np.isfinite(u):
            return False
        return margin_px <= u <= w - margin_px and margin_px <= v <= h - margin_px

    def direction(self, p: np.ndarray) -> np.ndarray:
        d = np.asarray(p, dtype=float) - self.pos
        return d / np.linalg.norm(d)

    def pixels_across(self, p: np.ndarray, diameter: float) -> float:
        """Apparent size of a sphere of `diameter` at `p`, in pixels."""
        _, _, depth = self.project(p)
        if depth <= 1e-9:
            return 0.0
        return float(self.cfg.focal_px[1] * diameter / depth)


def camera_pose(cfg: CameraConfig, table: TableConfig, side: str = "a") -> Pinhole:
    """Turn a side-local (back, height, lateral, pitch, yaw) into a world pose.

    Side 'a' sits behind the origin looking up +x; side 'b' is the mirror image
    through the centre line, so the same CameraConfig describes either mount.
    `lateral` and `yaw` are toward the camera's OWN left, which is +y for A and
    -y for B - so one positive `lateral` puts the two cameras on opposite sides
    of the centre line, which is exactly what triangulation needs.
    """
    side = side.lower()
    if side not in ("a", "b"):
        raise ValueError(f"side must be 'a' or 'b', not {side!r}")
    s = 1.0 if side == "a" else -1.0

    p = np.radians(cfg.pitch_deg)
    y = np.radians(cfg.yaw_deg)
    forward = np.array([s * np.cos(p) * np.cos(y), s * np.cos(p) * np.sin(y), -np.sin(p)])
    base_x = 0.0 if side == "a" else table.x_far
    pos = np.array([base_x - s * cfg.back, s * cfg.lateral, cfg.height])
    return Pinhole(pos=pos, rot=_basis(forward), cfg=cfg, name=f"cam_{side}")


def rig_poses(rig: CameraRig, table: TableConfig) -> tuple[Pinhole, Pinhole]:
    return camera_pose(rig.a, table, "a"), camera_pose(rig.b, table, "b")


# -- putting them in the scene ------------------------------------------------


def add_camera(spec: mujoco.MjSpec, cam: Pinhole, name: str | None = None):
    """Add one fixed camera with real intrinsics.

    Given `sensorsize` and `focal`, MuJoCo derives the frustum from the physical
    optics instead of from a hand-picked fovy, so swapping the lens in
    CameraConfig changes the render and the maths together.
    """
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(cam.rot.reshape(-1)))
    c = spec.worldbody.add_camera()
    c.name = name or cam.name
    c.pos = list(map(float, cam.pos))
    c.quat = list(map(float, quat))
    c.resolution = [int(v) for v in cam.cfg.resolution]
    c.sensor_size = [1e-3 * cam.cfg.sensor_mm[0], 1e-3 * cam.cfg.sensor_mm[1]]
    c.focal_length = [1e-3 * cam.cfg.focal_mm, 1e-3 * cam.cfg.focal_mm]
    return c


def add_cameras(
    spec: mujoco.MjSpec, table: TableConfig, rig: CameraRig | None = None
) -> mujoco.MjSpec:
    """Add `cam_a` and `cam_b`: one fixed camera behind each arm."""
    rig = rig or CameraRig()
    if not rig.enabled:
        return spec
    a, b = rig_poses(rig, table)
    add_camera(spec, a, CAM_A)
    add_camera(spec, b, CAM_B)
    return spec


# -- what a camera can see ----------------------------------------------------


def table_corners(table: TableConfig, net_height: float = 0.0) -> np.ndarray:
    """The eight points a camera has to contain to "see the whole table": the
    four corners of the surface and, if the net is up, the four top corners of
    the net band."""
    hw, ln = table.half_width, table.length
    pts = [(0.0, hw, 0.0), (0.0, -hw, 0.0), (ln, -hw, 0.0), (ln, hw, 0.0)]
    if net_height > 0.0:
        for sy in (hw, -hw):
            pts.append((0.5 * ln, sy, net_height))
            pts.append((0.5 * ln, sy, 0.0))
    return np.array(pts)


def coverage(cam: Pinhole, table: TableConfig, cell: float = 0.02) -> float:
    """Fraction of the table surface that falls inside the frame.

    A grid rather than the corners alone: a camera can hold all four corners and
    still lose a strip in the middle to a short focal length's aspect ratio.
    """
    nx = max(2, int(round(table.length / cell)))
    ny = max(2, int(round(table.width / cell)))
    xs = np.linspace(table.x_near, table.x_far, nx)
    ys = np.linspace(-table.half_width, table.half_width, ny)
    seen = 0
    for x in xs:
        for y in ys:
            if cam.in_frame(np.array([x, y, 0.0])):
                seen += 1
    return seen / (nx * ny)


def _sphere_samples(cam: Pinhole, centre: np.ndarray, radius: float) -> np.ndarray:
    """Centre plus four points on the ball's silhouette, as seen by `cam`.

    Occlusion is not binary in reality: a ball half behind a link is still
    detectable, and a ball whose centre happens to sit on a link edge is not.
    Sampling the silhouette turns it into a fraction.
    """
    d = cam.direction(centre)
    a = np.cross(d, WORLD_UP)
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(d, np.array([1.0, 0.0, 0.0]))
    a /= np.linalg.norm(a)
    b = np.cross(d, a)
    r = 0.9 * radius
    return np.array([centre, centre + r * a, centre - r * a, centre + r * b, centre - r * b])


def visible_fraction(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cam: Pinhole,
    centre: np.ndarray,
    radius: float,
    ball_body: int,
) -> float:
    """Fraction of five ball samples that are both in frame and unobstructed.

    Rays are cast from the camera with the ball's own body excluded, so a hit
    short of the ball means something else - an arm link, the net, the table
    edge - is in the way. This is the number the whole "mount it behind the arm"
    plan lives or dies on.
    """
    pts = _sphere_samples(cam, np.asarray(centre, dtype=float), radius)
    geomid = np.zeros(1, dtype=np.int32)
    clear = 0
    for p in pts:
        if not cam.in_frame(p):
            continue
        vec = p - cam.pos
        dist = float(np.linalg.norm(vec))
        hit = mujoco.mj_ray(
            model, data, cam.pos, vec / dist, None, 1, int(ball_body), geomid
        )
        if hit < 0.0 or hit >= dist - 1e-4:
            clear += 1
    return clear / len(pts)


# -- two cameras together -----------------------------------------------------


def parallax_deg(a: Pinhole, b: Pinhole, p: np.ndarray) -> float:
    """Angle between the two view rays at the point. 90 deg is ideal; near 0 or
    near 180 is the degenerate case where depth is unobservable."""
    ca = np.dot(a.direction(p), b.direction(p))
    return float(np.degrees(np.arccos(np.clip(-ca, -1.0, 1.0))))


def triangulation_sigma(
    cams: list[Pinhole], p: np.ndarray, noise_px: float = 0.5
) -> tuple[float, float]:
    """Position uncertainty of a point seen by several cameras, in metres.

    Each camera contributes a bearing measurement: it pins the point down across
    its own line of sight to about `depth * noise_px / f_px`, and not at all
    along it. Stacking those as an information matrix
    `J = sum (I - u u^T) / sigma_i^2` and inverting gives the error ellipsoid.
    Returns (worst axis, best axis) as one-sigma lengths - the ratio of the two
    is the conditioning, and it blows up when the rays are parallel or
    anti-parallel, i.e. when two cameras stare down the same axis at each other.
    """
    J = np.zeros((3, 3))
    for cam in cams:
        u = cam.direction(p)
        _, _, depth = cam.project(p)
        if depth <= 1e-6:
            continue
        sigma = depth * noise_px / cam.cfg.focal_px[1]
        J += (np.eye(3) - np.outer(u, u)) / (sigma * sigma)
    eig = np.linalg.eigvalsh(J)
    # Relative, not absolute: a single bearing leaves one axis free, and the
    # eigenvalue that should be zero comes back at rounding error, not at zero.
    if eig.max() <= 0.0 or eig.min() <= 1e-9 * eig.max():
        return (float("inf"), float(np.sqrt(1.0 / eig.max())) if eig.max() > 0 else float("inf"))
    s = np.sqrt(1.0 / eig)
    return (float(s.max()), float(s.min()))
