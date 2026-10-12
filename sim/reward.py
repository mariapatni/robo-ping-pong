"""The cost function: hit underhand, with the face, and land it in bounds.

Three requirements, in the order they gate each other:

  1. **Face contact.** The ball must be struck by one of the paddle's flat
     faces - either side counts - and not by the rim, the neck or the mount.
     Measured from the contact normal, not from intent.
  2. **Underhand.** The outward face normal must be open upward at contact, and
     the ball must actually leave with upward velocity. Overhand is penalised,
     not merely unrewarded.
  3. **In bounds.** The ball must land on the opponent's half of the table.
     Height is explicitly not rewarded: a high loop and a flat drive score the
     same if they both land in.

The asymmetry between 2 and "miss" is the delicate part, and §`StrokeWeights`
explains why the overhand penalty is bounded below the in-bounds bonus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class StrokeWeights:
    """Weights for the stroke and outcome terms.

    Overhand is meant to be severely uneconomic, and it is: a full overhand
    that lands in bounds scores about +2.6 against the +17.8 of a clean
    underhand on target, so it is roughly 7x worse.

    The one constraint that is NOT a free parameter is the floor. A successful
    overhand must still beat a miss. The moment `overhand` grows past
    `face_contact + in_bounds + target`, the cheapest way to dodge the penalty
    is to never swing - and the policy learns to stand still, a penalty
    creating the behaviour it was meant to discourage. So:

        worst scored overhand = face + in_bounds - overhand
                              = 1.0 + 10.0 - 12.0 = -1.0, clipped to 0.0
        typical overhand      = 1.0 + 10.0 + 3.0 - 11.4 = +2.6
        clean underhand       = 1.0 + 10.0 + 3.0 +  3.8 = +17.8

    Raising `overhand` further makes the floor bite and standing still becomes
    competitive. 12.0 is as hard as this can be pushed safely.
    """

    in_bounds: float = 10.0  # landing on the opponent's half
    underhand: float = 4.0  # full marks for a face-up strike
    overhand: float = 12.0  # subtracted for a fully face-down strike
    face_contact: float = 1.0  # struck with a flat face rather than the rim
    target: float = 3.0  # proximity to the aim point inside the opponent's half
    target_sigma: float = 0.15  # metres

    # Stroke credit earned by a ball that misses the table. Not zero, because
    # early in training the stroke shaping is what teaches the motion at all -
    # but small, because a shot that does not land is not a return. At 1.0 a
    # well-struck ball sailing off the end outscores a scrappy overhand that
    # actually lands, and the policy learns to prefer form over winning the
    # point. Pinned by test_the_outcome_ordering_is_the_design.
    out_of_bounds_stroke_scale: float = 0.25

    # Shaping, decayed away across the curriculum.
    approach: float = 1.0
    approach_sigma: float = 0.20

    # Costs that protect the hardware rather than shape the task.
    joint_velocity: float = 0.01
    action_rate: float = 0.01


def face_hit_quality(contact_normal, face_normal) -> float:
    """1.0 for a square face hit, 0.0 for a rim or edge hit.

    `abs` because both sides of the blade are legal: the paddle is symmetric
    and the policy may return with either face.
    """
    c = np.asarray(contact_normal, float)
    f = np.asarray(face_normal, float)
    nc, nf = np.linalg.norm(c), np.linalg.norm(f)
    if nc < 1e-9 or nf < 1e-9:
        return 0.0
    alignment = abs(float(c @ f) / (nc * nf))
    # Below 45 degrees of alignment the ball has caught the rim, not the face.
    return float(np.clip((alignment - 0.707) / (1.0 - 0.707), 0.0, 1.0))


def striking_normal(face_normal, contact_normal=None, ball_vel_out=None):
    """Which way the paddle actually pushed, as a unit vector.

    The blade is symmetric and `wrist_roll` can present either side, so the
    site's own normal is an arbitrary choice of face: rolled 180 degrees it
    points the other way while the stroke is identical. Taking it at face value
    scored a perfectly good lift off the back face as a full overhand.

    So the face is oriented by what the contact DID. The push direction is the
    change in ball velocity; failing that, the contact normal. Only if neither
    is available does the site normal stand on its own.
    """
    f = np.asarray(face_normal, float)
    n = np.linalg.norm(f)
    if n < 1e-9:
        return UP.copy()
    f = f / n

    push = None
    if ball_vel_out is not None and np.linalg.norm(ball_vel_out) > 1e-9:
        push = np.asarray(ball_vel_out, float)
    elif contact_normal is not None and np.linalg.norm(contact_normal) > 1e-9:
        push = np.asarray(contact_normal, float)
    if push is not None and float(f @ push) < 0.0:
        f = -f
    return f


def lift(face_normal, contact_normal=None, ball_vel_out=None) -> float:
    """How 'open upward' the striking face is: +1 straight up, -1 straight down.

    Signed, so the same number drives the underhand reward and the overhand
    penalty, with no discontinuity at horizontal.
    """
    return float(striking_normal(face_normal, contact_normal, ball_vel_out)[2])


def stroke_term(face_normal, ball_vel_out, w: StrokeWeights, contact_normal=None) -> float:
    """Reward an underhand strike, penalise an overhand one.

    The upward-velocity gate is on the *consequence*, not the pose: a policy
    can hold the paddle face-up and still chop down through the ball, and that
    should not score as underhand. Requiring the ball to actually leave with
    upward velocity is the thing a faked pose cannot satisfy.
    """
    l = lift(face_normal, contact_normal, ball_vel_out)
    v = np.asarray(ball_vel_out, float)
    going_up = float(v @ UP) > 0.2  # m/s, see docs/reward-design.md
    if l >= 0.0 and going_up:
        return w.underhand * l
    if l >= 0.0:
        return 0.0  # face-up but chopped down: neither rewarded nor punished
    return w.overhand * l  # l is negative here, so this subtracts


def landing_term(landing_xy, table, aim_xy=None, w: StrokeWeights | None = None) -> tuple[float, bool]:
    """Score where the ball landed. Returns (reward, in_bounds).

    Height is deliberately absent: a loop and a drive that land in the same
    place are worth the same. What the shot must do is land on the opponent's
    half, and preferably where that opponent can reach it.
    """
    w = w or StrokeWeights()
    x, y = float(landing_xy[0]), float(landing_xy[1])
    on_table = (table.x_net <= x <= table.x_far) and (abs(y) <= table.half_width)
    if not on_table:
        return 0.0, False
    r = w.in_bounds
    if aim_xy is not None:
        d = float(np.hypot(x - aim_xy[0], y - aim_xy[1]))
        r += w.target * float(np.exp(-d / w.target_sigma))
    return r, True


def exchange_reward(
    *,
    contact_normal=None,
    face_normal=None,
    ball_vel_out=None,
    landing_xy=None,
    table,
    aim_xy=None,
    weights: StrokeWeights | None = None,
) -> dict:
    """Score one exchange: the arm's single attempt to return one ball.

    A fault - no contact, a rim hit, or a ball that lands off the table - scores
    **zero, never negative**. If faults were negative the policy could improve
    its return by fouling deliberately (swatting the ball away with the elbow
    beats a slow, scored miss), which is reward hack 9 in the design doc.
    """
    w = weights or StrokeWeights()
    out = {"face": 0.0, "stroke": 0.0, "landing": 0.0, "total": 0.0,
           "contacted": False, "face_hit": False, "in_bounds": False}

    if contact_normal is None or face_normal is None:
        return out
    out["contacted"] = True

    quality = face_hit_quality(contact_normal, face_normal)
    out["face"] = w.face_contact * quality
    if quality <= 0.0:
        out["total"] = 0.0  # caught the rim: a fault, scores nothing
        return out
    out["face_hit"] = True

    if landing_xy is not None:
        out["landing"], out["in_bounds"] = landing_term(landing_xy, table, aim_xy, w)

    # Stroke credit is scaled down when the ball did not land in: the stroke is
    # shaping toward a return, and a ball that misses the table is not one.
    scale = 1.0 if out["in_bounds"] else w.out_of_bounds_stroke_scale
    raw = stroke_term(face_normal, ball_vel_out, w, contact_normal) * quality
    out["stroke"] = raw * (scale if raw > 0.0 else 1.0)

    out["total"] = max(0.0, out["face"] + out["stroke"] + out["landing"])
    return out
