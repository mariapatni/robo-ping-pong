"""A hard constraint: the arm cannot be commanded to drive the paddle into the table.

Not a reward term. A reward term is a price, and a price can be paid: the
optimiser weighs it against everything else and will violate the limit when the
payoff is large enough. arXiv 2505.22974 measured exactly this on a current
limit and reported that "the baseline policy without the constraint violated
the constraint even with soft over-current penalties", while the constrained
policy "never violated the constraint" on hardware.

A table strike is in that category. On the real arm the paddle does not pay a
price for entering the table, it hits it: the servo stalls against a surface it
cannot move, draws stall current, and the 3D-printed blade is what gives way.
A policy that learned the move in simulation - where a soft contact lets the
blade sink 24 mm in and keep going - would be learning something the hardware
cannot do, which is the definition of a transfer failure.

So the limit is enforced where it cannot be traded away: between the policy's
output and the servos. The policy proposes a joint target, this clamps it to
one that keeps the blade clear, and the clamped target is what reaches the bus.
The SAME layer runs on hardware, which is what makes it transfer - the policy
never experiences a world where the table is optional.

The cost is that the clamp is now part of the plant: the policy must learn
against the clamped dynamics, not the free ones. That is the right trade. It is
also why this is a projection onto the feasible set rather than a hard stop -
the arm keeps moving in the directions that are still legal.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class SafetyLimits:
    """What the arm is physically not allowed to do."""

    table_margin: float = 0.004  # metres of clearance to insist on
    bisect_steps: int = 7  # resolution of the projection, ~1% of the step


class TableGuard:
    """Clamps joint commands so the blade stays out of the table.

    Stateless apart from the model handle, so one guard serves any number of
    arms, and cheap enough to sit in a 50 Hz loop: the worst case is
    `bisect_steps` forward-kinematics evaluations of five joints.
    """

    def __init__(self, arm, limits: SafetyLimits | None = None, table_geom: str = "table"):
        self.arm = arm
        self.limits = limits or SafetyLimits()
        self._data = mujoco.MjData(arm.model)
        self.clamped = 0
        self.proposed = 0

        model = arm.model
        self.table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, table_geom)
        if self.table < 0:
            raise KeyError(f"no geom named {table_geom!r}")

        # Everything rigidly attached to the arm, which is what may not touch
        # the table. An earlier version modelled only the blade as a disc and
        # missed most of the problem: measured over 24 swings, the FOREARM
        # touched the table more often than the blade did (3,996 contact-steps
        # against 3,315), with the camera mount and wrist behind them. A
        # hand-written geometric check is the wrong tool - it can only ever
        # cover the shapes someone remembered. The collision detector already
        # knows every geom in the model.
        ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
        self.arm_geoms = {
            g for g in range(model.ngeom)
            if g not in (self.table, ball)
            and (model.geom_contype[g] or model.geom_conaffinity[g])
            and model.body_rootid[model.geom_bodyid[g]] != 0
        }

    def clearance(self, qpos) -> float:
        """Smallest gap between any arm geom and the table at this pose.

        Negative means interpenetration. Uses MuJoCo's own narrow-phase, so it
        is exact for the real geometry rather than an approximation of it.
        """
        d = self._data
        d.qpos[self.arm.qpos_adr] = qpos
        d.qvel[:] = 0.0
        mujoco.mj_kinematics(self.arm.model, d)
        mujoco.mj_collision(self.arm.model, d)
        worst = np.inf
        for c in range(d.ncon):
            con = d.contact[c]
            pair = (con.geom1, con.geom2)
            if self.table in pair and (set(pair) & self.arm_geoms):
                worst = min(worst, float(con.dist))
        return worst

    def is_safe(self, qpos) -> bool:
        return self.clearance(qpos) >= self.limits.table_margin

    def clamp(self, target, current):
        """The furthest point along current -> target that is still legal.

        Bisection rather than a solve: the clearance is monotone enough along a
        short joint-space step, and seven evaluations put the answer within
        about 1% of the step, which is far finer than the 4 mm margin.
        """
        target = np.asarray(target, float)
        current = np.asarray(current, float)
        self.proposed += 1
        if self.is_safe(target):
            return target

        self.clamped += 1
        if not self.is_safe(current):
            # Already in violation - the only safe move is not to make it worse.
            return current

        lo, hi = 0.0, 1.0  # lo is known safe, hi is known unsafe
        for _ in range(self.limits.bisect_steps):
            mid = 0.5 * (lo + hi)
            if self.is_safe(current + (target - current) * mid):
                lo = mid
            else:
                hi = mid
        return current + (target - current) * lo

    def command(self, qpos):
        """Drop-in replacement for `arm.command` that cannot foul the table."""
        safe = self.clamp(qpos, self.arm.qpos)
        return self.arm.command(safe)

    @property
    def clamp_rate(self) -> float:
        return self.clamped / max(1, self.proposed)
