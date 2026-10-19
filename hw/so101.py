"""The real SO-101, behind the same contract as sim.arm.Arm: five joint targets,
in radians, in ARM_JOINTS order.

A joint's angle is `sign * (ticks - center) * 2*pi/4096`. After hw/rehome.py
has run, every center is 2048 - the homing offset in the servo puts the middle
of each joint's mechanical range there, which is also where the sim puts zero
(its joint ranges are symmetric). Before that, the center is taken from the
mid-point of whatever angle limits are already in the servos' EEPROM.

Every move made through `move_to` is checked in the simulator first: the path
is sampled and refused if any arm geom would come within `clearance` of the
table. `stream` - used to mirror a policy - checks each target the same way
and holds the last legal one instead.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import feetech as ft

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = Path(__file__).resolve().parent / "calibration.json"
BACKUP = Path(__file__).resolve().parent / "calibration_backup.json"  # as the arm arrived
PORT = "/dev/cu.usbmodem5A680086121"

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
IDS = (1, 2, 3, 4, 5)
RAD_PER_TICK = 2 * np.pi / ft.TICKS
STALL_LOG = ROOT / "out" / "hw" / "stalls.jsonl"


class Stall(RuntimeError):
    """A joint stopped following its target."""


def load_calibration() -> dict | None:
    try:
        return json.loads(CALIBRATION.read_text())
    except OSError:
        return None


class SimCheck:
    """The simulator's table guard, used to vet real-arm targets."""

    def __init__(self, clearance: float = 0.010):
        from sim.env import EnvConfig, PingPongEnv
        from sim.paddle import PaddleConfig
        from sim.safety import SafetyLimits, TableGuard

        self.world = PingPongEnv(EnvConfig(paddle=PaddleConfig()))
        self.guard = TableGuard(self.world, SafetyLimits(table_margin=clearance))
        self.lower, self.upper = self.world.lower, self.world.upper

    def path_ok(self, q0, q1, n: int = 40) -> bool:
        """Every point clear of the table - or, if the arm already starts in
        violation (as it can before calibration), never getting worse."""
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        floor = min(self.guard.limits.table_margin, self.guard.clearance(q0) - 0.001)
        return all(self.guard.clearance(q0 + (q1 - q0) * s) >= floor for s in np.linspace(0, 1, n))


class SO101:
    def __init__(self, port: str = PORT, check: SimCheck | None = None, log=print):
        self.bus = ft.Bus(port)
        self.log = log
        missing = [i for i in IDS if not self.bus.ping(i)]
        if missing:
            raise ft.BusError(f"servos {missing} not answering")
        cal = load_calibration()
        if cal:
            self.fold_path = np.array(cal["_fold"], float) if "_fold" in cal else None
            self.center = np.array([cal[j]["center"] for j in JOINTS], float)
            self.sign = np.array([cal[j]["sign"] for j in JOINTS], float)
            self.lo = np.array([cal[j]["min"] for j in JOINTS], float)
            self.hi = np.array([cal[j]["max"] for j in JOINTS], float)
        else:
            # Not yet rehomed: the frame of the calibration the arm arrived with,
            # from the saved copy - never from live EEPROM, which a rehome
            # opens up while it runs.
            orig = json.loads(BACKUP.read_text())
            self.lo = np.array([orig[j]["min"] for j in JOINTS], float)
            self.hi = np.array([orig[j]["max"] for j in JOINTS], float)
            self.center = (self.lo + self.hi) / 2
            self.fold_path = None
            self.center[4] = 2048  # wrist roll has no stops
            self.sign = np.ones(5)
        self.check = check or SimCheck()
        if self.fold_path is None:
            # Old-calibration fold, tested on the arm; pan and roll left alone.
            q = self.q()
            self.fold_path = np.array([[q[0], -1.10, 1.64, -1.55, q[4]],
                                       [q[0], -0.60, 0.64, -1.20, q[4]],
                                       [q[0], -0.60, 0.64, 0.00, q[4]]])

    # The arm rests folded on its own base. Leaving or entering that fold by a
    # straight line drags the paddle over the table, so both go via the same
    # waypoints: rest <-> raised, paddle up <-> raised, paddle level. Shoulder
    # and elbow move together between rest and raised: with the shoulder
    # already tilted back, folding the elbow alone jams at a physical block.

    def unfold(self, speed: float = 0.25) -> None:
        for w in self.fold_path[1:]:
            self.move_to(w, speed=speed, check=False)

    def fold(self, speed: float = 0.25) -> None:
        q = self.q()
        up = self.fold_path[-1].copy()
        if np.abs(q - up).max() > 0.05:
            self.move_to(up, speed=speed)
        for w in self.fold_path[-2:0:-1]:
            self.move_to(w, speed=speed, check=False)
        self.move_to(self.fold_path[0], speed=0.4, check=False, tol=0.25)

    # -- units -----------------------------------------------------------------

    def to_rad(self, ticks) -> np.ndarray:
        return self.sign * (np.asarray(ticks, float) - self.center) * RAD_PER_TICK

    def to_ticks(self, q) -> np.ndarray:
        t = self.center + np.asarray(q, float) / RAD_PER_TICK * self.sign
        return np.clip(np.round(t), self.lo, self.hi).astype(int)

    # -- io ----------------------------------------------------------------------

    def ticks(self) -> np.ndarray:
        for i in range(3):
            try:
                p = self.bus.sync_read(ft.POSITION, IDS)
                return np.array([p[i] for i in IDS], float)
            except ft.BusError:
                if i == 2:
                    raise
        raise AssertionError

    def q(self) -> np.ndarray:
        return self.to_rad(self.ticks())

    def write_ticks(self, t) -> None:
        self.bus.sync_write(ft.GOAL_POSITION, {i: int(v) for i, v in zip(IDS, t)})

    def torque(self, on: bool) -> None:
        if on:
            # Hold where it is, so enabling torque cannot make it jump.
            self.write_ticks(self.ticks())
        for i in IDS:
            self.bus.write(i, ft.TORQUE_ENABLE, int(on))

    def hold(self) -> None:
        self.write_ticks(self.ticks())

    torque_cap = 1.0  # fraction of full output the servos are limited to

    def configure(self, acceleration: int = 0, torque_limit: int = 500) -> None:
        """RAM settings for motion. acceleration 0 = the servo's own maximum.
        The default caps output at 50%, so a collision is a push, not a crush;
        normal moves on this 5 V arm need under 35%."""
        self.torque_cap = torque_limit / 1000
        for i in IDS:
            self.bus.write(i, ft.ACCELERATION, acceleration)
            self.bus.write(i, ft.GOAL_SPEED, 0)
            self.bus.write(i, ft.TORQUE_LIMIT, torque_limit)

    # -- motion ----------------------------------------------------------------

    def safe_target(self, q) -> np.ndarray:
        return np.clip(np.asarray(q, float), self.check.lower, self.check.upper)

    def loads(self) -> np.ndarray:
        """Output duty per joint, -1..1. The servos cap themselves at 20%
        torque after 2 s above 80% (their overload protection)."""
        for i in range(3):
            try:
                ld = self.bus.sync_read(ft.LOAD, IDS)
                return np.array([ft.decode_sm(ld[i], 10) for i in IDS], float) / 1000
            except ft.BusError:
                if i == 2:
                    raise
        raise AssertionError

    def move_to(self, q, retries: int = 3, **kw) -> np.ndarray:
        """`_move_once`, retried from wherever the arm stopped if it stalls.

        This arm stalls intermittently: a joint sits at high output without
        moving, and the identical move, retried after a short release, goes
        through. Every stall is appended to out/hw/stalls.jsonl so the cause
        can be found later; a stall that survives every retry still raises.
        """
        for attempt in range(retries + 1):
            try:
                return self._move_once(q, **kw)
            except Stall as e:
                self.hold()
                with open(STALL_LOG, "a") as f:
                    f.write(json.dumps({"t": time.time(), "attempt": attempt, "why": str(e),
                                        "q": np.round(self.q(), 4).tolist(),
                                        "load": np.round(self.loads(), 3).tolist(),
                                        "target": np.round(np.asarray(q, float), 4).tolist()}) + "\n")
                if attempt == retries:
                    raise
                self.log(f"  stall ({e}); releasing and retrying {attempt + 1}/{retries}")
                time.sleep(0.4)
        raise AssertionError

    def _move_once(self, q, speed: float = 0.6, rate: float = 50.0, settle: float = 2.0,
                   tol: float = 0.15, check: bool = True, max_load: float = 0.75,
                   overload_s: float = 0.75) -> np.ndarray:
        """Interpolate to `q` at `speed` rad/s on the slowest joint, after
        checking the straight-line path in sim, then wait up to `settle` s for
        every joint to be within `tol` rad. Raises if it never gets there: a
        joint that cannot follow must not be mistaken for one that did."""
        q1 = np.asarray(q, float)
        q0 = self.q()
        if check and not self.check.path_ok(q0, q1):
            raise RuntimeError(f"refused: path {np.round(q0, 2)} -> {np.round(q1, 2)} meets the table in sim")
        n = max(1, int(np.ceil(np.abs(q1 - q0).max() / speed * rate)))
        heavy = 0  # cycles in a row with some joint above max_load
        peak = np.zeros(5)

        def watch():
            nonlocal heavy, peak
            ld = np.abs(self.loads())
            peak = np.maximum(peak, ld)
            heavy = heavy + 1 if ld.max() > min(max_load, 0.9 * self.torque_cap) else 0
            if heavy > overload_s * rate:
                self.hold()
                self.log(f"  loads at overload {np.round(ld, 2)}, q {np.round(self.q(), 2)}")
                raise Stall(f"overload: joint {JOINTS[int(ld.argmax())]} at {ld.max():.0%} "
                                   f"for {overload_s} s on the way to {np.round(q1, 2)} - holding")

        t0 = time.perf_counter()
        for k in range(1, n + 1):
            self.write_ticks(self.to_ticks(q0 + (q1 - q0) * k / n))
            watch()
            time.sleep(max(0.0, t0 + k / rate - time.perf_counter()))
        deadline = time.perf_counter() + settle
        while True:
            watch()
            err = np.abs(self.q() - self.to_rad(self.to_ticks(q1)))
            if err.max() < tol:
                self.log(f"  reached {np.round(q1, 2)}  peak load {np.round(peak, 2)}")
                return self.q()
            if time.perf_counter() > deadline:
                self.hold()
                raise Stall(f"did not reach {np.round(q1, 2)}: off by {np.round(err, 2)} rad")
            time.sleep(1 / rate)
