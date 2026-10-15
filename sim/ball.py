"""The ball: its geometry, its aerodynamics, and the scripted feeder.

Two things are worth knowing about what is *not* here. There is no spin in v1,
so there is no Magnus term - a 40 mm ball at 1-1.5 m/s with the topspin a rigid
PLA paddle can impart curves by a couple of centimetres over the table, which is
below the error in our table measurements. And drag is applied by us rather than
by MuJoCo, because MuJoCo's built-in `viscosity`/`density` fluid model is linear
(Stokes) plus a crude quadratic term, and at Re ~ 3000 the flow is squarely in
the quadratic regime where we want to name Cd ourselves.

Everything in this module is a pure function of (state, config) except the two
helpers that write into an MjData, and those write a whole array at once so the
port to MJX is a `.at[].set()`.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import BallConfig, LaunchConfig

BALL_BODY = "ball"
BALL_GEOM = "ball"


def add_ball(spec: mujoco.MjSpec, cfg: BallConfig, pos=(0.0, 0.0, 0.5)) -> mujoco.MjSpec:
    """Add a free-flying sphere with the real mass and radius.

    The mass is set explicitly rather than via density: a 2.7 g hollow ball is
    nothing like a solid 40 mm sphere of anything, and getting the mass wrong
    would make both the drag and the bounce wrong.

    The inertia is a thin shell's, 2/3 m r^2, not the solid sphere's 2/5 m r^2
    MuJoCo assumes by default. A real ball is ~1.7x harder to spin up than the
    solid one, which decides how much spin paddle friction puts on it in an
    off-centre or angled hit, and so how it comes off the rubber.
    """
    body = spec.worldbody.add_body(name=BALL_BODY, pos=list(pos))
    body.add_freejoint(name=BALL_BODY)
    body.add_geom(
        name=BALL_GEOM,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[cfg.radius, 0.0, 0.0],
        mass=cfg.mass,
        typeinertia=mujoco.mjtGeomInertia.mjINERTIA_SHELL,
        rgba=[1.0, 0.45, 0.05, 1.0],
        condim=3,
    )
    return spec


# -- aerodynamics -------------------------------------------------------------


def drag_force(vel: np.ndarray, drag_k: float) -> np.ndarray:
    """Quadratic aerodynamic drag, F = -0.5*rho*Cd*A*|v|*v = -drag_k*|v|*v.

    Written branch-free so it survives a port to a jitted step: no `if speed >
    0`, the zero-velocity case falls out of the multiply.
    """
    vel = np.asarray(vel, dtype=float)
    return -drag_k * np.linalg.norm(vel) * vel


# -- launcher -----------------------------------------------------------------


@dataclass(frozen=True)
class Launch:
    """A sampled feed: where the ball appears and how fast it is going."""

    pos: np.ndarray
    vel: np.ndarray

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))


def sample_launch(cfg: LaunchConfig, seed: int, table_length: float) -> Launch:
    """Sample one feed aimed from the far half back at the arm's half.

    Deterministic in `seed` alone: same seed, same feed, on any machine. The
    draws are taken in a fixed order so that adding a range later does not
    silently renumber the ones already in use.
    """
    rng = np.random.default_rng(seed)

    def u(rng_range: tuple[float, float]) -> float:
        return float(rng.uniform(*rng_range))

    pos = np.array([u(cfg.x_frac) * table_length, u(cfg.y), u(cfg.z)])
    speed = u(cfg.speed)
    elevation = np.radians(u(cfg.elevation_deg))
    # Azimuth 0 points straight down -x, i.e. at the arm.
    azimuth = np.pi + np.radians(u(cfg.azimuth_deg))
    horizontal = speed * np.cos(elevation)
    vel = np.array(
        [horizontal * np.cos(azimuth), horizontal * np.sin(azimuth), speed * np.sin(elevation)]
    )
    return Launch(pos=pos, vel=vel)


def drag_free_landing(launch: Launch, z_surface: float = 0.0, gravity: float = 9.81) -> np.ndarray:
    """Where the feed would first touch the surface with no drag.

    This is the closed-form check the tests compare against: if the simulated
    landing point is not within a few centimetres of it for a slow feed, either
    the drag or the units are wrong.
    """
    vz = launch.vel[2]
    h = launch.pos[2] - z_surface
    t = (vz + np.sqrt(vz * vz + 2.0 * gravity * h)) / gravity
    return np.array([launch.pos[0] + launch.vel[0] * t, launch.pos[1] + launch.vel[1] * t, z_surface])
