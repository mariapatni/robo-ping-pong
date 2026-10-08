"""Every physical number the simulation depends on, in one place.

The table is a kitchen island and the paddle does not exist yet, so these
numbers will change. They live in frozen dataclasses rather than as module
constants so that a test can ask for a variant ("same table, restitution 0.5")
without mutating global state, and so that a future JAX/MJX port can treat a
config as a static leaf rather than something read out of module scope.

World frame convention, fixed here and relied on everywhere else:

    +x  is the direction of play. The arm's base sits at x = 0, the middle of
        its own short edge; its half is x in [0, half_length].
    +y  is across the table, y = 0 on the centre line of play.
    z = 0 is the TABLE TOP, not the floor. The floor is at z = -height.

Keeping the arm base at the origin means sim/arm.py, the existing scripts and
the existing camera placements all keep working unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

GRAVITY = 9.81
AIR_DENSITY = 1.204  # kg/m^3, dry air at 20 C, sea level


@dataclass(frozen=True)
class TableConfig:
    """A kitchen island standing in for a table; dimensions are outer size.

    Measured 2026-09: 31.5 x 49.5 inches.
    """

    width: float = 0.800  # along y
    length: float = 1.257  # along x, i.e. the direction of play
    height: float = 0.900  # floor to playing surface
    thickness: float = 0.040  # slab depth below the playing surface

    # Ball-on-laminate, measured range 0.80-0.93; 0.88 is the middle.
    restitution: float = 0.88
    # Contact stiffness as an angular frequency, rad/s. This is the one number
    # that decides whether the bounce is right: the contact must last several
    # timesteps or the integrator invents energy (see contact_solref).
    # 1000 rad/s with a 0.5 ms step gives ~6 steps of contact and 1-3 mm of
    # penetration, and reproduces a requested restitution to about +/-0.02.
    contact_omega: float = 1000.0
    friction: tuple[float, float, float] = (0.25, 0.005, 0.0001)

    @property
    def half_length(self) -> float:
        """Depth of one player's half. The arm owns x in [0, half_length]."""
        return 0.5 * self.length

    @property
    def half_width(self) -> float:
        return 0.5 * self.width

    @property
    def contact_damping_ratio(self) -> float:
        """Damping ratio of the contact spring that yields `restitution`.

        A linear spring-damper rebounds with exp(-pi*z/sqrt(1-z^2)); inverting
        that gives z. It is only an approximation of what the solver does, so
        the achieved restitution is measured in the tests, not trusted.
        """
        e = min(max(self.restitution, 1e-3), 0.999)
        r = -math.log(e) / math.pi
        return r / math.sqrt(1.0 + r * r)

    @property
    def contact_solref(self) -> tuple[float, float]:
        """MuJoCo solref in its explicit (-stiffness, -damping) form.

        The default (timeconst, dampratio) form is expressed relative to the
        step size and, for a 2.7 g ball, was wildly unstable at the low damping
        ratios a bouncy contact needs. Naming the spring directly is stable and
        maps onto restitution in a way we can check.
        """
        w = self.contact_omega
        return (-w * w, -2.0 * self.contact_damping_ratio * w)


@dataclass(frozen=True)
class BallConfig:
    """A regulation 40 mm table tennis ball."""

    radius: float = 0.020
    mass: float = 0.0027

    # Quadratic drag: F = -0.5 * rho * Cd * A * |v| * v.
    # A 40 mm ball at 1-1.5 m/s sits at Re ~ 1.5e3-4e3, where a smooth sphere's
    # Cd is just under 0.5; the table-tennis literature uses 0.4-0.5 for
    # non-spinning flight, so 0.47 is the middle of both.
    drag_coefficient: float = 0.47
    air_density: float = AIR_DENSITY

    @property
    def area(self) -> float:
        return float(np.pi * self.radius**2)

    @property
    def drag_k(self) -> float:
        """Precomputed 0.5*rho*Cd*A so the step loop is one multiply."""
        return 0.5 * self.air_density * self.drag_coefficient * self.area

    @property
    def terminal_speed(self) -> float:
        """Sanity handle: ~7 m/s. Rally speeds are far below it, which is why
        drag shows up as a few percent per metre rather than as a hard cap."""
        return float(np.sqrt(self.mass * GRAVITY / self.drag_k))


@dataclass(frozen=True)
class LaunchConfig:
    """Randomisation ranges for the scripted feeder.

    The ball is spawned on the far half and aimed back at the arm's half at
    rally speed. Ranges are (low, high), sampled uniformly. The x range is
    expressed as a fraction of table length so it survives the next
    re-measurement of the island.
    """

    x_frac: tuple[float, float] = (0.88, 0.98)  # spawn depth, far half
    y: tuple[float, float] = (-0.15, 0.15)
    z: tuple[float, float] = (0.12, 0.22)  # above the table top
    speed: tuple[float, float] = (1.0, 1.5)  # deliberately a slow rally
    elevation_deg: tuple[float, float] = (-2.0, 8.0)  # + is upwards
    azimuth_deg: tuple[float, float] = (-6.0, 6.0)  # 0 is straight down -x


@dataclass(frozen=True)
class SurfaceConfig:
    """How the playing surface looks. Waxed wood, not table-tennis green.

    The island is oak-toned waxed wood, and the vision stack has to work on it:
    a ball detector tuned against a flat green slab would be learning a
    background that does not exist. The grain is generated procedurally (see
    `sim.table.wood_texture`) rather than shipped as a PNG so there is no binary
    asset to keep in sync with the numbers here, and so the palette can be
    re-measured against a photograph later by editing two tuples.
    """

    wood: bool = True
    texture_px: int = 512
    seed: int = 7

    # Measured-by-eye off waxed oak: the pale field and the dark grain line.
    light_rgb: tuple[float, float, float] = (0.76, 0.58, 0.36)
    dark_rgb: tuple[float, float, float] = (0.42, 0.26, 0.13)
    grain_lines: float = 7.0  # dark lines per texture tile, across the grain
    grain_contrast: float = 2.2  # >1 thins the dark lines into streaks
    planks: int = 4  # seams along the length, as on a butcher-block island

    # texrepeat in (along x, across y) tiles. The grain runs along +x, the
    # direction of play, which is how a plank island is actually laid.
    repeat: tuple[float, float] = (2.0, 1.0)

    # Wax, i.e. a broad low-intensity highlight rather than a mirror.
    specular: float = 0.35
    shininess: float = 0.25
    reflectance: float = 0.08

    # Used when wood=False, so a test can ask for the old flat slab.
    plain_rgba: tuple[float, float, float, float] = (0.10, 0.35, 0.28, 1.0)


@dataclass(frozen=True)
class MarkerConfig:
    """The four corner markers that define the bounds of play.

    They are also the stand-in for the calibration fiducials that will be taped
    to the real island, which is why each corner gets its own colour: a camera
    that can only see three of them must still know which three. Non-colliding
    on purpose - a ball clipping a sticker is not a physical event worth
    simulating, and a collidable disc on the surface would change the bounce.
    """

    enabled: bool = True
    radius: float = 0.030
    thickness: float = 0.0015
    inset: float = 0.045  # from each edge, so the whole marker is on the table

    # STag fiducials, not plain discs. STag refines its homography against an
    # inner circular border, which is what makes its pose steadier than a
    # square-only marker - and every millimetre of error in the table frame
    # lands directly in the ball position the policy sees.
    fiducial: bool = True
    hd: int = 21  # 12 markers, maximum error correction; we need four
    ids: tuple[int, int, int, int] = (0, 1, 2, 3)
    size: float = 0.070  # printed edge length, metres
    texture_px: int = 256

    # Near-left, near-right, far-right, far-left, going round the table.
    # Distinct hues, and distinct in greyscale too: the OV9281 is monochrome.
    rgba: tuple[tuple[float, float, float, float], ...] = (
        (0.95, 0.15, 0.15, 1.0),  # near +y  (arm A's left) - luma 0.39
        (0.98, 0.95, 0.15, 1.0),  # near -y                  - luma 0.87
        (0.15, 0.85, 0.30, 1.0),  # far  -y  (arm B's left)  - luma 0.58
        (0.10, 0.25, 0.85, 1.0),  # far  +y                  - luma 0.28
    )


@dataclass(frozen=True)
class NetConfig:
    """The centre divider at x = length/2. DO NOT "fix" this back to 152.5 mm.

    Height: a regulation net is 152.5 mm on a 2740 x 1525 mm table, i.e. 5.6% of
    the length or 10.0% of the width. Scaled onto this 1257 x 800 mm island that
    would be 70 mm or 80 mm - but nothing else scales with the table. The ball is
    still 40 mm and the SO-101's usable strike band is z in [0.03, 0.22] m, so an
    80 mm net eats a third of the vertical window the arm has to work in.

    A regulation net is not merely disproportionate here, it is geometrically
    infeasible, and that is measured twice, independently:

      * Forward: integrating real returns (Cd = 0.47, e = 0.88) from a realistic
        contact at x = 0.356, z = 0.095 over the whole sustainable outgoing
        envelope - 1.55-1.85 m/s, below which the ball bounces on its own half
        and above ~2.1 m/s the second bounce leaves the table - puts the ball
        over the midline at 29-123 mm.
      * Backward: simulating the feeds LaunchConfig actually produces puts them
        over the midline at 18-134 mm (median 76). See
        tests/test_scene.py::test_a_regulation_net_would_block_every_feed_we_have.

    So a 152.5 mm net blocks every legal shot this table can support, in both
    directions of play. 60 mm - within the 50-70 mm band both estimates leave
    open - is 1.5 ball diameters, still tall enough to force a genuine arc and to
    be the visible centre reference for calibration, and it clears the upper end
    of both envelopes.

    `collide` is off by default, and that is also a measurement rather than
    laziness: even at 60 mm, a colliding net rejects roughly a third of the feeds
    LaunchConfig currently produces, so switching it on without retuning the
    feeder would silently delete part of the training distribution. Net contact
    is a rules concept before it is a physics one, and in v1 it is better checked
    as a midline clearance predicate (`z >= 0.030` at x = length/2) than paid for
    with a rigid geom standing in for a compliant mesh. The flag exists so the
    switch is one line once the feeder serves over the net; when it is on, the
    net carries its own very lossy contact so a clipped ball drops instead of
    rebounding off a wall.
    """

    enabled: bool = True
    height: float = 0.060
    thickness: float = 0.004
    overhang: float = 0.012  # past each side edge, as a real net does
    collide: bool = False

    restitution: float = 0.10
    contact_omega: float = 300.0

    rgba: tuple[float, float, float, float] = (0.92, 0.92, 0.95, 0.55)
    post_radius: float = 0.006
    post_rgba: tuple[float, float, float, float] = (0.12, 0.12, 0.14, 1.0)

    @property
    def contact_damping_ratio(self) -> float:
        e = min(max(self.restitution, 1e-3), 0.999)
        r = -math.log(e) / math.pi
        return r / math.sqrt(1.0 + r * r)

    @property
    def contact_solref(self) -> tuple[float, float]:
        w = self.contact_omega
        return (-w * w, -2.0 * self.contact_damping_ratio * w)


@dataclass(frozen=True)
class CameraConfig:
    """One fixed camera, behind and above its own arm, angled down.

    Wrist cameras are rejected, so each side gets a single fixed USB camera and
    the two are fused by one Kalman filter in the table frame. The pose is given
    in the camera's OWN side frame, so the same numbers describe both sides:

        back     metres behind its own arm's base, away from the table
        height   metres above the table top
        lateral  metres toward its own left (+y for arm A, -y for arm B)
        pitch    degrees below horizontal
        yaw      degrees off the centre line, toward its own left

    Optics are given as a real sensor and a real lens rather than as a MuJoCo
    fovy, because the thing we have to buy is a lens: an OV9281 is 3.84 x 2.40 mm
    of active area, and MuJoCo will derive the frustum from that plus the focal
    length exactly as the physical camera does.
    """

    # Defaults are the winner of scripts/camera_study.py over 12 launcher
    # seeds: all four table corners in frame on a 1.8 mm lens, 11 px across the
    # ball, 72 deg median parallax between the pair, 2.0 mm triangulated
    # position sigma, and a worst-case fused blackout of two 40 ms frames.
    # The yaw is negative because the lateral offset is positive: the camera
    # steps toward its own left corner and then looks back across the table,
    # which is what keeps the opposite near corner in frame.
    back: float = 0.15
    height: float = 0.60
    lateral: float = 0.25
    pitch_deg: float = 50.0
    yaw_deg: float = -10.0

    focal_mm: float = 1.8
    sensor_mm: tuple[float, float] = (3.84, 2.40)
    resolution: tuple[int, int] = (640, 400)

    @property
    def fov_h_deg(self) -> float:
        return 2.0 * math.degrees(math.atan(0.5 * self.sensor_mm[0] / self.focal_mm))

    @property
    def fov_v_deg(self) -> float:
        return 2.0 * math.degrees(math.atan(0.5 * self.sensor_mm[1] / self.focal_mm))

    @property
    def focal_px(self) -> tuple[float, float]:
        """Focal length in pixels, per axis. Equal for a square-pixel sensor."""
        w, h = self.resolution
        return (self.focal_mm * w / self.sensor_mm[0], self.focal_mm * h / self.sensor_mm[1])


@dataclass(frozen=True)
class CameraRig:
    """The pair. Two configs, not one, so an asymmetric rig is expressible.

    Both are written in their own side frame, so a non-zero `lateral` puts the
    two cameras on opposite sides of the centre line in world coordinates -
    which is the point: two cameras on the centre line looking down +x and -x at
    each other see the ball along one shared axis and cannot triangulate its
    depth at all.
    """

    enabled: bool = True
    a: CameraConfig = field(default_factory=CameraConfig)
    b: CameraConfig = field(default_factory=CameraConfig)

    @staticmethod
    def symmetric(cam: CameraConfig) -> "CameraRig":
        return CameraRig(a=cam, b=cam)


@dataclass(frozen=True)
class EnvConfig:
    table: TableConfig = field(default_factory=TableConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    launch: LaunchConfig = field(default_factory=LaunchConfig)
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    markers: MarkerConfig = field(default_factory=MarkerConfig)
    net: NetConfig = field(default_factory=NetConfig)
    cameras: CameraRig = field(default_factory=CameraRig)

    # The vendored model integrates at 200 Hz, far too coarse to resolve a stiff
    # 2.7 g contact: at 200 Hz the bounce spans one step and the integrator
    # injects energy, sending the ball metres into the air. 2 kHz gives ~6 steps
    # of contact and a restitution that tracks the configured value.
    timestep: float = 0.0005

    def with_restitution(self, e: float) -> "EnvConfig":
        return replace(self, table=replace(self.table, restitution=e))
