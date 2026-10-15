"""The keep-up env counts real bounces, and only real bounces.

The reward is +1 per bounce, so the bounce counter IS the objective, and the
ways it can be gamed are the ways it is tested: a ball resting on the blade
must not score, a held paddle must score the few bounces physics gives it for
free, and nothing the policy asks for may put the arm into the table.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.keepup import KeepUpConfig, KeepUpEnv  # noqa: E402
from sim.paddle import face_pose  # noqa: E402


@pytest.fixture(scope="module")
def env():
    e = KeepUpEnv()
    yield e
    e.close()


def run(env, policy, seed=0):
    obs, _ = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        total += r
        done = term or trunc
    return info, total


def test_nominal_pose_holds_the_face_level(env):
    env.world.reset(env.q0)
    centre, normal = face_pose(env.world.model, env.world.data)
    assert normal[2] > 0.999
    assert np.allclose(centre, (0.30, 0.0, 0.18), atol=0.005)


def test_spaces(env):
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (33,)
    assert env.action_space.shape == (5,)


def test_held_paddle_earns_the_free_bounces_then_loses_the_ball(env):
    """Measured: ~4 bounces before the ball dies. If this jumps, the contact
    model or the bounce counter changed; if it reads 0, contacts are missed."""
    bounces = [run(env, lambda o: np.zeros(5), seed=s)[0]["bounces"] for s in range(5)]
    assert 2 <= np.mean(bounces) <= 6


def test_a_ball_resting_on_a_still_blade_does_not_score():
    """No reset noise, so the arm starts exactly on its target and stays still.

    With noise it does not: the step from the noisy pose to q0 sets the arm
    ringing at ~21 Hz with little damping (the vendored STS3215 gains carrying
    the paddle), and a ringing blade really does throw a resting ball 5 cm up.
    That is a correct bounce, so it is not what this test is about.
    """
    env = KeepUpEnv(KeepUpConfig(joint_noise=0.0))
    env.reset(seed=0)
    face, _ = face_pose(env.world.model, env.world.data)
    env.world.set_ball(face + np.array([0.0, 0.0, env.r + 0.001]))
    env.armed = False  # it has not risen from anywhere
    info = {}
    for _ in range(int(3.0 / env.dt)):
        _, _, term, trunc, info = env.step(np.zeros(5))
        if term or trunc:
            break
    env.close()
    assert info["bounces"] == 0
    assert info.get("cause") == "stalled"


def test_no_action_puts_the_arm_in_the_table(env):
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    for _ in range(150):
        env.step(rng.choice([-1.0, 1.0], 5))  # bang-bang: the worst case
        assert env.guard.clearance(env.world.qpos) > -0.002


def test_the_default_world_is_the_same_every_episode(env):
    """v1 was trained without randomisation; evaluating it must stay possible."""
    m = env.world.model
    seen = set()
    for s in range(4):
        env.reset(seed=s)
        seen.add((env.substeps, float(m.actuator_gainprm[env.world.actuator_ids[0], 0]),
                  float(m.body_mass[env.world.ball_body])))
    assert len(seen) == 1


def test_randomized_world_differs_between_episodes_and_stays_in_range():
    env = KeepUpEnv(KeepUpConfig.randomized())
    R, m = env.cfg.rand, env.world.model
    mass0, kp0 = env._ball_mass, env._kp.copy()
    draws = []
    for s in range(6):
        env.reset(seed=s)
        mass = float(m.body_mass[env.world.ball_body])
        kp = m.actuator_gainprm[env.world.actuator_ids, 0]
        hz = 1.0 / env.dt
        assert R.ball_mass[0] - 1e-9 <= mass / mass0 <= R.ball_mass[1] + 1e-9
        assert np.all((kp / kp0 >= R.servo_kp[0] - 1e-9) & (kp / kp0 <= R.servo_kp[1] + 1e-9))
        assert R.control_hz[0] - 2 <= hz <= R.control_hz[1] + 2  # rounding to whole substeps
        draws.append((round(mass, 7), env.substeps))
    env.close()
    assert len(set(draws)) > 1


def test_observation_noise_does_not_touch_the_true_state():
    env = KeepUpEnv(KeepUpConfig.randomized())
    env.reset(seed=0)
    true_ball = env.world.ball_pos()
    a, b = env._obs(), env._obs()
    env.close()
    assert not np.allclose(a, b)  # noisy
    assert np.allclose(env.world.ball_pos(), true_ball)  # observing changed nothing


def _flight(env, rise, angle, seconds, landed=None):
    """Score one flight with the given apex, takeoff angle off ideal, and
    duration, coming down at `landed` (the centre by default)."""
    env.reset(seed=0)
    face_z = 0.18
    env.takeoff = (0.0, face_z, angle, angle)
    env.apex_z = face_z + env.r + rise
    return env._score_flight(seconds, env.home if landed is None else np.asarray(landed))


def test_a_high_straight_bounce_beats_a_fast_low_dribble(env):
    """The score is per second of flight, so frequency alone cannot win.

    v1 was paid per bounce and learned an ~8 cm dribble at ~4 bounces a second.
    Over the same second of play, a 0.27 m straight bounce must now be worth
    several times more than that dribble.
    """
    g = 9.81
    t_high = 2 * np.sqrt(2 * 0.27 / g)
    t_low = 2 * np.sqrt(2 * 0.08 / g)
    per_s_high = _flight(env, 0.27, 0.0, t_high) / t_high
    per_s_low = _flight(env, 0.08, 0.0, t_low) / t_low
    assert per_s_high > 3 * per_s_low


def test_straight_up_beats_slanted_and_too_high_is_not_free(env):
    t = 0.45
    straight = _flight(env, 0.27, 0.0, t)
    slanted = _flight(env, 0.27, 25.0, t)
    too_high = _flight(env, 0.60, 0.0, t)
    assert straight > 1.6 * slanted
    assert straight > 2.5 * too_high


def test_coming_down_at_the_centre_beats_coming_down_away_from_it(env):
    home = _flight(env, 0.27, 0.0, 0.45)
    away = _flight(env, 0.27, 0.0, 0.45, landed=env.home + np.array([0.15, 0.0]))
    assert home > 1.8 * away


def test_off_centre_the_ideal_bounce_leans_back_towards_the_centre(env):
    """At the centre the ideal is vertical; 10 cm out, a vertical bounce is no
    longer ideal, and one leaning home is."""
    env.reset(seed=0)
    up = np.array([0.0, 0.0, 2.3])
    at_home = np.array([*env.home, 0.2])
    out = np.array([env.home[0] + 0.10, env.home[1], 0.2])
    assert env._takeoff_angles(at_home, up)[0] < 1e-6
    flight = 2 * up[2] / 9.81
    lean_home = np.array([-0.5 * 0.10 / flight, 0.0, up[2]])
    lean_away = lean_home * np.array([-1.0, 1.0, 1.0])
    assert env._takeoff_angles(out, lean_home)[0] < 1e-6
    # The lean is small - 2.6 degrees carries a 0.27 m bounce 5 cm - so most of
    # the pull home comes from where the ball lands, not from this angle.
    assert env._takeoff_angles(out, up)[0] < env._takeoff_angles(out, lean_away)[0]


def test_catching_the_drop_earns_the_flat_bonus(env):
    env.reset(seed=0)
    assert env.takeoff is None
    assert env._score_flight(1.0, env.home) == env.cfg.catch


def test_drops_come_from_the_footprint_and_the_curriculum_narrows_them():
    env = KeepUpEnv(KeepUpConfig.randomized())
    fp = env.footprint
    assert len(fp) > 100
    r = np.hypot(fp[:, 0], fp[:, 1])
    assert 0.15 < r.min() and r.max() < 0.47  # a horseshoe around the base
    assert fp[:, 1].min() < -0.3 and fp[:, 1].max() > 0.3  # it reaches both sides
    far = []
    for s in range(40):
        env.reset(seed=s)
        far.append(np.hypot(*(env.world.ball_pos()[:2] - env.home)))
    env.spawn_scale = 0.1
    near = []
    for s in range(40):
        env.reset(seed=s)
        near.append(np.hypot(*(env.world.ball_pos()[:2] - env.home)))
    env.close()
    assert max(far) > 0.2  # well beyond the blade, but only as far as is catchable
    assert max(near) < 0.1


def test_every_drop_is_one_the_arm_can_get_under_in_time():
    """The first v3 run spent 18M steps on drops 30-36 cm out that the arm
    needs ~0.4 s to reach and a 25-60 cm drop gives it 0.23-0.35 s for."""
    env = KeepUpEnv(KeepUpConfig.randomized())
    fp, need = env.footprint, env.reach_time
    for s in range(150):
        env.reset(seed=s)
        ball = env.world.ball_pos()
        if env.spawn_dist < 0.03:
            continue  # a centre drop: the arm is already there
        nearest = np.argmin(np.hypot(*(fp - ball[:2]).T))
        fall = np.sqrt(2 * (ball[2] - env.r - env.face0[2]) / 9.81)
        assert need[nearest] <= fall + 1e-9
    env.close()


def test_a_zero_action_holds_the_arm_still(env):
    env.reset(seed=0)
    target = env.target.copy()
    for _ in range(10):
        env.step(np.zeros(5))
    assert np.allclose(env.target, target)


def test_one_step_moves_a_target_no_faster_than_the_servo_can(env):
    env.reset(seed=0)
    before = env.target.copy()
    env.step(np.ones(5))
    assert np.all(np.abs(env.target - before) <= env.cfg.max_joint_speed * env.dt + 1e-9)


def test_curriculum_stage_zero_drops_every_ball_at_one_spot():
    env = KeepUpEnv(KeepUpConfig.randomized())
    env.spawn_scale = 0.0
    spots = []
    for s in range(30):
        env.reset(seed=s)
        spots.append(env.world.ball_pos()[:2] - env.home)
    env.close()
    assert np.abs(np.array(spots)).max() <= env.cfg.drop_offset + 1e-9


def test_staying_in_the_game_pays_more_than_a_beginner_spends_moving():
    """The first v3 run learned to end episodes early: every step only cost.

    A beginner's actions are noisy (training starts at std ~0.37). Over an
    episode its movement costs must not outweigh the reward for keeping the
    ball alive, or knocking the ball away becomes the best strategy.
    """
    env = KeepUpEnv(KeepUpConfig(joint_noise=0.0))
    c, rng = env.cfg, np.random.default_rng(0)
    env.reset(seed=0)
    env.world.set_ball(env.world.park_pos)  # keep the ball out of it
    costs, prev = [], np.zeros(5)
    for _ in range(90):
        a = np.clip(rng.normal(0.0, 0.37, 5), -1, 1)
        env.step(a)
        costs.append(c.action_rate * np.sum((a - prev) ** 2)
                     + c.joint_velocity * np.sum(env.world.qvel ** 2))
        prev = a
    env.close()
    assert np.mean(costs) < 0.5 * c.alive * env.dt


def test_a_curriculum_stage_practises_its_newest_ring():
    env = KeepUpEnv(KeepUpConfig.randomized())
    env.spawn_scale_prev, env.spawn_scale = 0.3, 0.4  # ring 0.18-0.24 m out
    dists = []
    for s in range(200):
        env.reset(seed=s)
        dists.append(env.spawn_dist)
    env.close()
    d = np.array(dists)
    in_ring = np.mean((d > 0.18) & (d <= 0.24 + 0.03))  # + jitter of half a grid step
    assert in_ring > 0.35  # half of the 75% off-centre, plus its share of the rest
    assert d.max() <= 0.24 + 0.03


def test_takeoff_scatter_turns_the_ball_without_changing_its_speed(env):
    env.reset(seed=0)
    v = np.array([0.0, 0.0, 2.0])
    turned = [env._scatter(v, 2.0) for _ in range(200)]
    speeds = np.linalg.norm(turned, axis=1)
    angles = np.degrees(np.arccos(np.clip(np.array(turned) @ v / (speeds * 2.0), -1, 1)))
    assert np.allclose(speeds, 2.0)
    assert 1.0 < np.sqrt(np.mean(angles ** 2)) < 3.5
