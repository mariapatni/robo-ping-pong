"""The whole stroke: where to start, where to aim, and when to go.

Kept in one place because the three parts are coupled and were previously
spread across the scripts, which is how they drifted apart. A swing is:

  wind-up   drawn back along the hitting direction, clear of the table
  through   a point BEYOND the ball, not the ball itself
  timing    the ramp started so the paddle passes the ball at the right moment

The third one is the part that is easy to get wrong in two different ways, and
both were measured:

A swing that ends at the ball is decelerating when it arrives. The paddle
delivered 0.14 m/s where the plan asked for 0.70 - a fifth of the intended
energy - because it was at a third of its own peak speed at contact. That is
what put 37% of returns back on the arm's own half.

But simply swinging harder is worse, not better. Holding the ramp duration
fixed and extending the path made the paddle 75% faster and dropped returns
from 42% to 33%: it hits harder and less accurately. So the follow-through
lengthens the path AND the time together, keeping the speed the planner asked
for and moving the deceleration to after contact rather than before it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pose import PoseSolution, solve_face_pose_ik, swing_through, windup_pose

SWING_TIME = 0.34  # seconds from wind-up to contact


@dataclass(frozen=True)
class Swing:
    wind: PoseSolution
    strike: PoseSolution
    through: PoseSolution
    lead: float  # seconds before contact that the ramp starts
    duration: float  # seconds for the whole ramp, wind -> through
    follow: float  # metres travelled past the ball

    def command_at(self, t: float, contact_time: float) -> np.ndarray:
        """Joint target at time `t`, for a contact scheduled at `contact_time`."""
        start = contact_time - self.lead
        if t < start:
            return self.wind.qpos
        a = min(1.0, (t - start) / self.duration)
        return self.wind.qpos + (self.through.qpos - self.wind.qpos) * a


def plan_swing(arm, guard, strike_point, face_normal, paddle_speed: float,
               blade_radius: float = 0.055, swing_time: float = SWING_TIME,
               seed: int = 0) -> Swing:
    """Build the stroke around a strike point and the face it needs.

    `paddle_speed` sets how far the follow-through runs: a gentle block barely
    travels past the ball, a firmer return travels further. Not every stroke
    should follow through the same amount - a long follow-through on a soft
    return overshoots and leaves the paddle somewhere useless for the next
    exchange.
    """
    strike_point = np.asarray(strike_point, float)
    n = np.asarray(face_normal, float)
    n = n / max(np.linalg.norm(n), 1e-9)

    hit = solve_face_pose_ik(arm, strike_point, n, seed=seed, guard=guard)
    wind_pt = windup_pose(strike_point, n, blade_radius=blade_radius)
    wind = solve_face_pose_ik(arm, wind_pt, n, seed=seed, guard=guard)
    thru_pt, follow = swing_through(strike_point, n, paddle_speed, swing_time)
    thru = solve_face_pose_ik(arm, thru_pt, n, seed=seed, guard=guard)

    # Where along the wind -> through path the ball sits, in joint space. The
    # ramp runs at the speed the planner wanted, so a longer path simply takes
    # proportionally longer, and the extra length is spent AFTER contact.
    span = float(np.linalg.norm(thru.qpos - wind.qpos))
    to_hit = float(np.linalg.norm(hit.qpos - wind.qpos))
    frac = float(np.clip(to_hit / max(span, 1e-9), 0.2, 0.95))
    duration = swing_time / frac
    return Swing(wind=wind, strike=hit, through=thru, lead=swing_time,
                 duration=duration, follow=follow)
