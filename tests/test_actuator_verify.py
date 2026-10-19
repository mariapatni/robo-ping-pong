"""The actuator verifier's analysis, with no arm attached.

The verifier's job is to turn a recording into numbers the sim can be fitted
to, so the analysis is tested against recordings with known answers: a
synthetic actuator with a chosen delay, time constant and speed limit must be
measured and fitted back to those values.
"""

import numpy as np
import pytest

from hw.actuator_verify import (
    GRID_DT,
    Recording,
    build_programs,
    fit_model,
    lag_and_rms,
    replay_in_sim,
    simulate_model,
    step_metrics,
    unsafe_targets,
)
from sim.keepup import KeepUpEnv

DELAY, TAU, VMAX = 0.07, 0.03, 1.5


@pytest.fixture(scope="module")
def env():
    return KeepUpEnv()


def synthetic(cmd_t, cmd_q, j=2, noise=0.0, seed=0) -> Recording:
    """A recording made by the simple actuator model itself, read at ~2 kHz."""
    read_t = np.arange(0.0, cmd_t[-1] + 0.8, 0.0005)
    probe = Recording(cmd_t, cmd_q, read_t, np.tile(cmd_q[0], (len(read_t), 1)))
    y = simulate_model(probe, j, DELAY, TAU, VMAX)[:, 0]
    q = np.tile(cmd_q[0], (len(read_t), 1))
    q[:, j] = np.interp(read_t, probe.grid(), y) + np.random.default_rng(seed).normal(0, noise, len(read_t))
    return Recording(cmd_t, cmd_q, read_t, q)


def steps(q0, j=2, amp=0.25, n=6, hold=0.8):
    t = np.arange(n) * hold
    q = np.tile(q0, (n, 1))
    q[1::2, j] += amp
    return t, q


def test_fit_recovers_a_known_actuator(env):
    t, q = steps(env.q0)
    p = fit_model([synthetic(t, q)], 2)
    assert p["delay_s"] == pytest.approx(DELAY, abs=0.011)
    assert p["tau_s"] == pytest.approx(TAU, rel=0.5)
    assert p["vmax"] == pytest.approx(VMAX, rel=0.3)
    assert p["rms_deg"] < 0.3


def test_step_metrics_see_the_delay_and_the_speed_limit(env):
    t, q = steps(env.q0)
    ms = step_metrics(synthetic(t, q), 2)
    assert len(ms) == len(t) - 1
    dead = np.mean([m["dead_ms"] for m in ms])
    assert DELAY * 1000 < dead < DELAY * 1000 + 30  # delay, plus a little rise to 5%
    assert np.mean([m["speed"] for m in ms]) == pytest.approx(VMAX, rel=0.2)


def test_noise_shows_up_as_spread_not_as_a_shifted_mean(env):
    t, q = steps(env.q0, n=12)
    clean = step_metrics(synthetic(t, q), 2)
    noisy = step_metrics(synthetic(t, q, noise=np.radians(0.3)), 2)
    assert np.mean([m["t90_ms"] for m in noisy]) == pytest.approx(np.mean([m["t90_ms"] for m in clean]), abs=15)


def test_lag_of_a_tracked_sine_is_found(env):
    t = np.arange(0, 6, 1 / 30)
    q = np.tile(env.q0, (len(t), 1))
    q[:, 2] += 0.03 * np.sin(2 * np.pi * 0.8 * t)
    lag, rms = lag_and_rms(synthetic(t, q), 2)
    assert DELAY - 0.01 < lag < DELAY + TAU + 0.05
    assert rms < 0.01


def test_the_default_programs_are_safe_and_a_bad_one_is_refused(env):
    segs = build_programs(env.q0, ["rest", "steps", "chirp"], range(5), repeats=2)
    assert unsafe_targets(segs, env.world.lower, env.world.upper, env.guard) == []
    segs[1].q[3] = env.world.upper + 0.5
    assert unsafe_targets(segs, env.world.lower, env.world.upper, env.guard)


def test_the_sim_arm_replays_a_recording(env):
    t, q = steps(env.q0, n=4)
    rec = synthetic(t, q)
    sim = replay_in_sim(rec, env.world)
    assert sim.shape == (len(rec.grid()), 5)
    # The MuJoCo servo is fast: by the end of each hold it sits on the command.
    end = np.searchsorted(rec.grid(), t[1] + 0.7)
    assert sim[end, 2] == pytest.approx(q[1, 2], abs=0.02)
    assert GRID_DT > 0
