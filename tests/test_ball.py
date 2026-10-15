"""The ball falls, bounces as hard as we asked it to, and is slowed by air.

Same shape as tests/test_arm.py: set a state, run the physics, check the number
that came back. These are the tests that will catch it if a MuJoCo upgrade
quietly changes how contacts behave, which for a 2.7 g ball is the difference
between a rally and a ball in orbit.
"""

import sys
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.ball import BALL_BODY, Launch, drag_force, drag_free_landing, sample_launch  # noqa: E402
from sim.config import GRAVITY, BallConfig, EnvConfig  # noqa: E402
from sim.env import PingPongEnv  # noqa: E402
from sim.reach import reach_map  # noqa: E402


@pytest.fixture(scope="module")
def env():
    return PingPongEnv()


def drop(env: PingPongEnv, height: float, seconds: float = 1.5) -> dict:
    """Drop the ball from `height` above the table and report the first bounce.

    Restitution is read off the velocity either side of the contact, which is
    its definition, and off the rebound apex, which is what you can see.
    """
    env.reset()
    env.set_ball((0.35 * env.cfg.table.length, 0.0, height))
    n = int(seconds / env.model.opt.timestep)
    z = np.empty(n)
    vz = np.empty(n)
    in_contact = np.empty(n, dtype=bool)
    for i in range(n):
        env.step(1)
        z[i] = env.ball_pos()[2]
        vz[i] = env.ball_vel()[2]
        in_contact[i] = env.data.ncon > 0

    touch = np.flatnonzero(in_contact)
    assert touch.size, "ball never touched the table"
    a = int(touch[0])
    b = a
    while b + 1 < n and in_contact[b + 1]:
        b += 1
    r = env.cfg.ball.radius
    apex = z[b:].max()
    return {
        "e_velocity": float(-vz[b + 1] / vz[a - 1]),
        "e_apex": float(np.sqrt(max(apex - r, 0.0) / (height - r))),
        "apex": float(apex),
        "min_z": float(z.min()),
    }


# -- drag, as a pure function -------------------------------------------------


def test_drag_force_matches_the_equation():
    cfg = BallConfig()
    v = np.array([1.2, -0.5, 0.3])
    expected = -0.5 * cfg.air_density * cfg.drag_coefficient * cfg.area * np.linalg.norm(v) * v
    assert np.allclose(drag_force(v, cfg.drag_k), expected)


def test_drag_opposes_motion_and_is_quadratic():
    k = BallConfig().drag_k
    v = np.array([1.0, 0.0, 0.0])
    assert np.allclose(drag_force(np.zeros(3), k), 0.0)
    assert drag_force(v, k)[0] < 0.0
    # Doubling the speed quadruples the force.
    assert np.allclose(drag_force(2 * v, k), 4 * drag_force(v, k))


# -- bouncing -----------------------------------------------------------------


def test_ball_falls_and_bounces(env):
    out = drop(env, 0.30)
    assert out["min_z"] < 0.02, "ball never reached the table"
    assert out["min_z"] > 0.0, "ball tunnelled through the table"
    assert out["apex"] > 0.10, "ball did not bounce back up"


@pytest.mark.parametrize("restitution", [0.60, 0.80, 0.88, 0.93])
def test_bounce_matches_configured_restitution(restitution):
    env = PingPongEnv(EnvConfig().with_restitution(restitution))
    out = drop(env, 0.30)
    assert out["e_velocity"] == pytest.approx(restitution, abs=0.06)
    assert out["e_apex"] == pytest.approx(restitution, abs=0.06)


def test_bouncier_table_bounces_higher(env):
    dull = drop(PingPongEnv(EnvConfig().with_restitution(0.60)), 0.30)["apex"]
    lively = drop(PingPongEnv(EnvConfig().with_restitution(0.93)), 0.30)["apex"]
    assert lively > dull + 0.05


# -- drag, in the simulation --------------------------------------------------


def run_horizontal(cfg: EnvConfig, speed: float, seconds: float) -> np.ndarray:
    """Launch horizontally from high above the table and return the final
    velocity, stopping before the ball can reach the surface."""
    env = PingPongEnv(cfg)
    env.reset()
    env.set_ball((0.1, 0.0, 0.70), (speed, 0.0, 0.0))
    env.step(int(seconds / env.model.opt.timestep))
    assert env.ball_pos()[2] > 0.05, "ball hit the table; shorten the run"
    return env.ball_vel()


def integrate_drag(cfg: EnvConfig, vel0, seconds: float, steps: int = 20000) -> np.ndarray:
    """Reference solution of dv/dt = g - (k/m)|v|v by RK4, independent of MuJoCo.

    The closed form for quadratic drag only exists in 1-D, and a falling ball is
    not 1-D: after a third of a second it is dropping faster than it is flying,
    so the vertical speed dominates |v| and couples into the horizontal decay.
    Integrating the real ODE is the only honest comparison.
    """
    k_over_m = cfg.ball.drag_k / cfg.ball.mass
    g = np.array([0.0, 0.0, -GRAVITY])

    def a(v):
        return g - k_over_m * np.linalg.norm(v) * v

    v = np.asarray(vel0, dtype=float)
    h = seconds / steps
    for _ in range(steps):
        k1 = a(v)
        k2 = a(v + 0.5 * h * k1)
        k3 = a(v + 0.5 * h * k2)
        k4 = a(v + h * k3)
        v = v + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return v


def test_drag_slows_a_horizontal_launch():
    speed, seconds = 1.25, 0.35
    cfg = EnvConfig()
    with_drag = run_horizontal(cfg, speed, seconds)
    without = run_horizontal(replace(cfg, ball=replace(cfg.ball, drag_coefficient=0.0)), speed, seconds)

    assert without[0] == pytest.approx(speed, abs=1e-6), "no-drag run should keep its x velocity"
    lost = without[0] - with_drag[0]
    assert lost > 0.02, f"drag is not measurable: lost only {lost * 1000:.1f} mm/s"

    reference = integrate_drag(cfg, (speed, 0.0, 0.0), seconds)
    assert with_drag[0] == pytest.approx(reference[0], rel=0.02)
    assert with_drag[2] == pytest.approx(reference[2], rel=0.02)


def test_terminal_speed_is_far_above_rally_speed():
    """If this ever drops near 1.5 m/s the drag coefficient or the mass is
    wrong, and every trajectory in the project is wrong with it."""
    assert 5.0 < BallConfig().terminal_speed < 10.0


# -- the launcher -------------------------------------------------------------


def test_launcher_is_reproducible(env):
    a = env.launch(1234)
    state_a = (env.ball_pos(), env.ball_vel())
    env.reset()
    b = env.launch(1234)
    assert np.allclose(a.pos, b.pos) and np.allclose(a.vel, b.vel)
    assert np.allclose(state_a[0], env.ball_pos()) and np.allclose(state_a[1], env.ball_vel())

    c = sample_launch(env.cfg.launch, 1234, env.cfg.table.length)
    assert np.allclose(a.pos, c.pos) and np.allclose(a.vel, c.vel)


def test_different_seeds_give_different_feeds(env):
    feeds = [env.launch(s) for s in range(8)]
    speeds = np.array([f.speed for f in feeds])
    assert len(np.unique(np.round(speeds, 6))) == len(feeds)


@pytest.mark.parametrize("seed", range(12))
def test_launch_respects_its_ranges_and_aims_at_the_arm(seed):
    cfg = EnvConfig()
    lc, t = cfg.launch, cfg.table
    feed = sample_launch(lc, seed, t.length)

    assert lc.x_frac[0] * t.length <= feed.pos[0] <= lc.x_frac[1] * t.length
    assert lc.y[0] <= feed.pos[1] <= lc.y[1]
    assert lc.z[0] <= feed.pos[2] <= lc.z[1]
    assert lc.speed[0] <= feed.speed <= lc.speed[1]
    assert feed.pos[0] > t.half_length, "feed should start on the far half"
    assert feed.vel[0] < 0.0, "feed should be heading at the arm"


@pytest.mark.parametrize("seed", [0, 3, 7, 11])
def test_landing_matches_the_drag_free_estimate(seed):
    """At 1-1.5 m/s over ~0.5 m of flight, drag costs a few percent of speed, so
    the textbook parabola should still be right to a couple of centimetres. If
    this test fails wide, suspect the units before the physics."""
    env = PingPongEnv()
    env.reset()
    feed = env.launch(seed)
    predicted = drag_free_landing(feed)

    n = int(1.5 / env.model.opt.timestep)
    landing = None
    for _ in range(n):
        env.step(1)
        if env.data.ncon > 0:
            landing = env.ball_pos()
            break
    assert landing is not None, "feed never reached the table"
    assert landing[0] == pytest.approx(predicted[0], abs=0.05)
    assert landing[1] == pytest.approx(predicted[1], abs=0.02)


def test_drag_free_landing_is_the_textbook_parabola():
    feed = Launch(pos=np.array([1.0, 0.0, 0.20]), vel=np.array([-1.2, 0.0, 0.0]))
    # Dropped from 0.2 m with no vertical velocity: t = sqrt(2h/g) = 0.2019 s.
    assert drag_free_landing(feed)[0] == pytest.approx(1.0 - 1.2 * np.sqrt(2 * 0.20 / 9.81), abs=1e-6)


# -- the table, and whether the arm can use it --------------------------------


def test_table_top_is_at_z_zero(env):
    """A ball set down on the surface stays there. With z = 0 as the table top,
    resting means a centre height of exactly one radius."""
    t = env.cfg.table
    env.reset()
    env.set_ball((t.half_length, 0.0, env.cfg.ball.radius))
    env.step(int(0.4 / env.model.opt.timestep))
    assert env.ball_pos()[2] == pytest.approx(env.cfg.ball.radius, abs=3e-3)
    assert env.tip_pos()[2] > 0.0, "the arm should sit on the table, not in it"


def test_ball_off_the_side_falls_to_the_floor(env):
    """Cheap check that the table is finite and the floor moved down with it."""
    t = env.cfg.table
    env.reset()
    env.set_ball((t.half_length, t.half_width + 0.10, 0.05))
    env.step(int(1.2 / env.model.opt.timestep))
    assert env.ball_pos()[2] < -0.5 * t.height


def test_arm_reaches_a_useful_part_of_its_half(env):
    """Coarse grid on purpose - this is a regression guard, not the measurement.
    scripts/reach.py prints the real number and the footprint."""
    rm = reach_map(env, env.cfg.table, samples=(25, 13, 13, 11, 3))
    assert rm.fraction > 0.40, f"only {rm.fraction:.1%} of the half is reachable"
    assert rm.max_x > 0.35, f"deepest reach only {rm.max_x:.3f} m"


@pytest.mark.parametrize("seed", [0, 5, 9])
def test_feeds_arrive_inside_the_arms_reach(seed):
    """The whole point of the launcher ranges: a feed the arm cannot get to is
    not a training signal, it is noise."""
    env = PingPongEnv()
    env.reset()
    env.launch(seed)
    reachable_depth, band = 0.45, (0.03, 0.22)
    arrived = False
    for _ in range(int(2.5 / env.model.opt.timestep)):
        env.step(1)
        p = env.ball_pos()
        if p[0] <= reachable_depth and band[0] <= p[2] <= band[1] and abs(p[1]) < 0.30:
            arrived = True
            break
        if p[2] < -0.1:
            break
    assert arrived, "feed never entered the arm's strike zone"


def test_ball_is_a_hollow_shell_not_a_solid_sphere(env):
    """2/3 m r^2, not MuJoCo's solid-sphere default of 2/5 m r^2."""
    m = env.model
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, BALL_BODY)
    mass, r = m.body_mass[body], env.cfg.ball.radius
    assert mass == pytest.approx(0.0027)
    assert np.allclose(m.body_inertia[body], 2.0 / 3.0 * mass * r * r, rtol=1e-6)


def test_the_fast_step_flies_the_ball_exactly_as_drag_force_says(env):
    """PingPongEnv.step inlines drag on plain floats for speed. Over a launch
    with bounces, it must track a step built on `drag_force` itself."""
    env.reset()
    env.set_ball((0.6, 0.0, 0.3), (-1.5, 0.3, 1.0))
    ref = mujoco.MjData(env.model)
    ref.qpos[:], ref.qvel[:] = env.data.qpos, env.data.qvel
    k, b, a = env.cfg.ball.drag_k, env.ball_body, env.ball_dof_adr
    for _ in range(2000):  # 1 s, through a bounce
        env.step(1)
        ref.ctrl[:] = env.data.ctrl
        ref.xfrc_applied[b, :3] = drag_force(ref.qvel[a:a + 3], k)
        mujoco.mj_step(env.model, ref)
    assert np.allclose(env.data.qpos, ref.qpos, atol=1e-9)
