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
class EnvConfig:
    table: TableConfig = field(default_factory=TableConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    launch: LaunchConfig = field(default_factory=LaunchConfig)

    # The vendored model integrates at 200 Hz, far too coarse to resolve a stiff
    # 2.7 g contact: at 200 Hz the bounce spans one step and the integrator
    # injects energy, sending the ball metres into the air. 2 kHz gives ~6 steps
    # of contact and a restitution that tracks the configured value.
    timestep: float = 0.0005

    def with_restitution(self, e: float) -> "EnvConfig":
        return replace(self, table=replace(self.table, restitution=e))
