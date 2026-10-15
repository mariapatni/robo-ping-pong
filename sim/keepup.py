"""Keep-up: one arm keeps a ball bouncing on its paddle, anywhere it can reach.

The first task a policy is trained on, and deliberately not the real one. A
return needs perception of a fast incoming ball, a swing timed inside a 178 ms
window and a landing 0.6 m away; keep-up needs only the part every one of those
rests on - getting the face under a falling ball, at the right time, moving the
right way. It is small enough to train on a laptop CPU and it exercises the
same plant: the same arm, paddle, contact model, 30 Hz control rate and table
guard the rally will use.

The task, in full: a ball is dropped from anywhere above the arm's face-up
footprint (`face_up_footprint`). The arm gets under it, keeps it bouncing,
walks it back to the centre of its workspace, and bounces it straight up there
for as long as it can.

What makes it a learning problem rather than a hold-still problem was measured
before anything was trained: a paddle held perfectly level keeps about 0.67 of
the apex height per bounce. So a dropped ball dies in five or six bounces unless
the arm puts energy back in on every contact, and drifts off the blade unless it
tilts to correct it.

The interface is the project's one contract: five joint position targets. The
policy moves those targets at up to `max_joint_speed` - a velocity command, so
it cannot ask the servos for a speed they do not have, and a zero action holds
the arm still. Every target goes through `TableGuard` before it reaches the
actuators, exactly as it will on hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import gymnasium as gym
import mujoco
import numpy as np

from .ball import BALL_GEOM
from .config import GRAVITY, EnvConfig
from .env import PingPongEnv
from .paddle import FACE_SITE, PADDLE_BODY, PaddleConfig, face_pose
from .reward import face_hit_quality
from .safety import TableGuard

# The centre of the workspace, where the arm starts and where it should bring
# the ball: 0.30 m out on the centre line, face 0.18 m above the table. It sits
# in the deepest part of the face-up footprint, so the arm has room to move
# under a ball that drifts in any direction.
NOMINAL_FACE = (0.30, 0.0, 0.18)


@dataclass(frozen=True)
class Randomization:
    """What changes from one episode to the next, beyond where the ball starts.

    Drawn once per episode at reset, except where noted. The reward and the
    termination checks always use the TRUE state; only what the policy observes
    is corrupted. The point is a policy that does not depend on any one number
    we are unsure of, because on hardware every one of these is wrong by some
    amount we will only measure later.
    """

    # Physics, as multipliers on (or ranges for) the nominal values.
    ball_mass: tuple[float, float] = (0.9, 1.1)
    drag_coefficient: tuple[float, float] = (0.40, 0.55)  # the literature's range
    paddle_restitution: tuple[float, float] = (0.78, 0.90)  # nominal 0.85, rubber unmeasured
    # ...and per bounce on top of that: no two hits on real rubber come off the
    # same, and the blade is deader towards its rim than at the sweet spot.
    # Redrawn at the start of every paddle contact. Without it the policy gets a
    # perfectly consistent paddle for 20 s and can learn to rely on that.
    restitution_jitter: float = 0.02  # one-sigma, per contact
    rim_loss: float = 0.08  # restitution lost at the blade's edge, quadratic in radius
    # ...and it does not come off quite where the contact model sends it:
    # rubber seams, dust, a little spin and blade flex. Every takeoff direction
    # is turned by this much, one-sigma, about a random horizontal axis.
    takeoff_scatter_deg: float = 2.0
    # The vendored STS3215 gains leave the arm ringing after a step, which the
    # real servos may or may not do. Wide damping range so the policy cannot
    # come to rely on the ringing, or on its absence.
    servo_kp: tuple[float, float] = (0.8, 1.2)
    servo_kv: tuple[float, float] = (0.6, 2.0)
    joint_damping: tuple[float, float] = (0.7, 1.3)

    # The decision rate, held near the 30 Hz the servo bus will actually run.
    # v2 drew it from 30-60 Hz without telling the policy, and came out a 60 Hz
    # specialist: 58.6 bounces at 60 Hz, 5.6 at 30. So the range is now the
    # bus's own jitter around 30, and the period is an observation.
    control_hz: tuple[float, float] = (27.0, 33.0)

    # Sensing, one-sigma. Ball position at the 2.0 mm the camera study
    # measured for the fused pair; joints at about one encoder tick (4096/rev).
    ball_pos_noise: float = 0.002
    ball_vel_noise: float = 0.05
    joint_pos_noise: float = 0.0015
    joint_vel_noise: float = 0.02
    # Camera pipeline lag: the ball is observed this many decisions late.
    ball_latency_steps: tuple[int, ...] = (0, 1)

    # Disturbances: a sideways nudge to the ball in flight, standing in for
    # spin, air currents and everything else the v1 ball model leaves out.
    push_rate_hz: float = 0.5
    push_speed: float = 0.15  # one-sigma, m/s, horizontal


@dataclass(frozen=True)
class KeepUpConfig:
    episode_s: float = 20.0
    # How fast the policy may move each joint's target, rad/s. About the
    # STS3215's no-load top speed, so nothing asked for here is beyond the
    # hardware - and enough to cross the footprint while a dropped ball falls.
    max_joint_speed: float = 4.5
    # Time a servo takes to start following a moved target. With the speed
    # above, it sets how long the arm needs to get under a ball - see
    # `reach_time` and `face_up_footprint`.
    servo_lag_s: float = 0.10

    # A bounce only scores if the ball rose at least this far above the face
    # since the last one that scored. Without it, the cheapest policy is to
    # rest the ball on the blade and buzz it: dozens of contacts a second,
    # none of them a bounce.
    min_rise: float = 0.04

    # Getting under the dropped ball at all. The first contact of an episode
    # ends a flight the policy did not launch, so it earns this flat bonus
    # instead of a flight score - which matters most for far drops, where the
    # catch is the hard part.
    catch: float = 1.0

    # What a good bounce is. Scored once per completed flight - takeoff from the
    # paddle to the next scoring contact - as
    #
    #     flight_weight * flight_time * height * straight * home
    #
    # Paying per SECOND of flight rather than per bounce is the point. v1 was
    # paid +1 a bounce inside a 20 s cap, so the best policy was the fastest
    # legal dribble: ~80 bounces at an ~8 cm apex, barely over `min_rise`. Per
    # second, a low bounce earns nothing for being frequent; it has to be good.
    #
    # Height: full marks for an apex 0.20-0.35 m above the face it left. 0.20 m
    # is 0.40 s in the air, twelve decisions at 30 Hz - time to correct a bad
    # bounce. 0.35 m keeps impact under 2.7 m/s and the drift from a small tilt
    # error inside the blade. Outside the band the score falls off smoothly, and
    # never below `low_credit`, so the first clumsy bounces still teach.
    flight_weight: float = 2.0  # reward per second of flight at full quality
    height_band: tuple[float, float] = (0.20, 0.35)
    height_sigma: float = 0.05
    low_credit: float = 0.25
    # Straight: scored on the takeoff velocity against the IDEAL direction,
    # which is not always vertical. Away from the centre, the ideal bounce
    # carries the ball `homing_gain` of the way back towards it over one
    # flight; at the centre that is straight up. Half credit is kept for any
    # angle so height still pays alone.
    straight_sigma_deg: float = 10.0
    homing_gain: float = 0.5
    # Home: the flight score is scaled by how near the centre it came down,
    # from half at a distance to full within a few centimetres.
    home_sigma: float = 0.06
    # A small bonus per contact for striking near the blade's centre.
    centre: float = 0.2
    centre_sigma: float = 0.025  # metres off the blade centre
    # Staying in the game, per second the ball is alive. Without it every
    # step only costs (the two terms below), and the first v3 run learned the
    # obvious answer: take the catch bonus and knock the ball away, ending the
    # episode in 0.45 s - 97% of episodes, return pinned at the catch bonus.
    # 0.5/s is small next to a good bounce (up to 2/s) but outweighs what a
    # still-clumsy policy spends moving, so leaving early never pays. A ball
    # resting on the blade cannot farm it: `stall_s` ends that episode.
    alive: float = 0.5
    # Costs that protect the hardware rather than shape the task. Kept small:
    # with velocity actions an exploring policy moves a lot, and at the old
    # 0.02 / 0.001 they summed to ~-25 an episode.
    action_rate: float = 0.005
    joint_velocity: float = 0.0002

    # Failure conditions.
    stall_s: float = 2.0  # no scoring bounce for this long: the ball is dead
    max_reach: float = 0.50  # the ball has gone further than this from the base

    # Where the ball starts. `spawn_radius` 0 drops it over the centre,
    # `drop_offset` off; above 0, anywhere in the face-up footprint within
    # `spawn_radius * spawn_scale` of the centre. `spawn_scale` belongs to the
    # env, not the config, so a curriculum can raise it as the policy earns it.
    # Even at full spread `centre_fraction` of drops stay at the centre, so the
    # policy keeps practising the rally while it learns to chase.
    spawn_radius: float = 0.0
    centre_fraction: float = 0.25
    # While a curriculum stage is being learned, this fraction of the
    # off-centre drops come from its newest ring - between `spawn_scale_prev`
    # and `spawn_scale` of the radius - so the stage is actually practised
    # rather than diluted by every distance already mastered.
    frontier_fraction: float = 0.5
    drop_height: tuple[float, float] = (0.08, 0.25)  # above the nominal face
    drop_offset: float = 0.015
    drop_speed: float = 0.05  # horizontal, m/s
    drop_vz: tuple[float, float] = (0.0, 0.0)  # m/s, thrown down (-) or up (+)
    joint_noise: float = 0.03

    # None: a fixed world. See `randomized()`.
    rand: Randomization | None = None

    @classmethod
    def randomized(cls, **overrides) -> "KeepUpConfig":
        """Drops from anywhere in the footprint, plus every `Randomization` term.

        Drop heights start at 0.25 m so a ball at the edge of the footprint
        falls for long enough (~0.25 s) for the arm to get there.
        """
        base = dict(spawn_radius=0.60, drop_height=(0.25, 0.60), drop_speed=0.10,
                    drop_vz=(-0.3, 0.2), joint_noise=0.06, rand=Randomization())
        base.update(overrides)
        return cls(**base)


class KeepUpEnv(gym.Env):
    """Gymnasium env: keep the ball bouncing, and bring it home.

    Observation (33,): joint angles and commanded targets relative to the
    nominal pose, joint velocities, the ball relative to the face centre and
    its velocity, the face normal, the face relative to the workspace centre,
    the previous action, and the control period (in units of 1/30 s).
    Ground-truth ball state - the camera stack is not in the loop yet; when it
    is, this is where its estimate goes.

    Action (5,): joint target velocities, as a fraction of `max_joint_speed`.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, cfg: KeepUpConfig | None = None, render_size: tuple[int, int] | None = None):
        self.cfg = cfg or KeepUpConfig()
        self.world = PingPongEnv(EnvConfig(paddle=PaddleConfig()), offscreen=render_size)
        self.guard = TableGuard(self.world)
        m = self.world.model

        self.q0 = nominal_pose(self.world, self.guard)
        self.substeps = self.world.steps_per_command  # redrawn per episode under `rand`
        self.dt = self.substeps * m.opt.timestep
        self.max_steps = int(round(self.cfg.episode_s / self.dt))
        self.latency = 0
        self.home = np.asarray(NOMINAL_FACE[:2], float)
        # 0..1: what fraction of `spawn_radius` drops may come from. Training
        # starts it at 0 - one spot - and raises it as the policy earns it;
        # evaluation leaves it at 1.
        self.spawn_scale = 1.0
        self.spawn_scale_prev = 1.0  # inner edge of the newest ring; see `frontier_fraction`
        self.spawn_dist = 0.0
        self.footprint, self.reach_time = None, None
        if self.cfg.spawn_radius > 0:
            self.footprint, poses = face_up_footprint(self.world, self.guard, q0=self.q0)
            # How long the arm needs to get the face from the centre to each
            # footprint point: its slowest joint at full speed, plus servo lag.
            self.reach_time = (np.abs(poses - self.q0).max(axis=1) / self.cfg.max_joint_speed
                               + self.cfg.servo_lag_s)

        paddle = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PADDLE_BODY)
        self.paddle_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == paddle}
        self.ball_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, BALL_GEOM)
        self.table_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
        self.arm_geoms = self.guard.arm_geoms - self.paddle_geoms
        self.face_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, FACE_SITE)
        self.r = self.world.cfg.ball.radius

        # Nominal values of everything `Randomization` touches, so each episode
        # draws from the same centre instead of drifting from the last draw.
        self._ball_cfg = self.world.cfg.ball
        self._ball_mass = float(m.body_mass[self.world.ball_body])
        self._ball_inertia = m.body_inertia[self.world.ball_body].copy()
        self._paddle_cfg = self.world.cfg.paddle
        act = self.world.actuator_ids
        self._kp = m.actuator_gainprm[act, 0].copy()
        self._kv = -m.actuator_biasprm[act, 2].copy()
        self._damping = m.dof_damping[self.world.qvel_adr].copy()

        self.world.reset(self.q0)
        self.face0, _ = face_pose(m, self.world.data)
        self.target = self.q0.copy()

        self.action_space = gym.spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (33,), np.float32)

        self._renderer = None
        self._render_size = render_size

    # -- gym API --------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c, rng = self.cfg, self.np_random
        self._randomize_world()
        q = self.q0 + rng.uniform(-c.joint_noise, c.joint_noise, 5)
        self.world.reset(q)
        self.target = self.world.qpos

        vxy = rng.uniform(-c.drop_speed, c.drop_speed, 2)
        vz = rng.uniform(*c.drop_vz)
        spawn = self._spawn_point()
        self.spawn_dist = float(np.hypot(*(spawn[:2] - self.home)))
        self.world.set_ball(spawn, (vxy[0], vxy[1], vz))
        self._seen_ball = (self.world.ball_pos(), self.world.ball_vel())

        self.t = 0
        self.bounces = 0
        self.prev_action = np.zeros(5)
        self.armed = True  # the drop itself counts as having risen
        self.in_contact = False
        self.since_bounce = 0.0
        self.peak = -np.inf  # highest the ball has risen above the face this flight
        self.cause = None
        self.takeoff = None  # (time, face z, angle off ideal, angle off vertical) of this flight
        self.apex_z = -np.inf
        self.flights = []  # (apex above face, angle off vertical) of every scored flight
        self.contacts_xy = []  # where each scoring contact happened
        return self._obs(), {}

    def step(self, action):
        c, w = self.cfg, self.world
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        want = w.clip(self.target + a * c.max_joint_speed * self.dt)
        self.target = self.guard.clamp(want, w.qpos)
        w.command(self.target)

        if c.rand and c.rand.push_rate_hz > 0 and not self.in_contact \
                and self.np_random.random() < c.rand.push_rate_hz * self.dt:
            v = self.world.ball_vel()
            v[:2] += self.np_random.normal(0.0, c.rand.push_speed, 2)
            self.world.set_ball(self.world.ball_pos(), v)

        reward = 0.0
        for _ in range(self.substeps):
            w.step(1)
            reward += self._contacts()
            if self.cause:
                break
            # Rise is measured against the face where it actually is, not the
            # nominal pose: a paddle lifted 2 cm would otherwise hand out half
            # of `min_rise` for free.
            z = w.data.qpos[w.ball_qpos_adr + 2]
            self.peak = max(self.peak, z - self.r - w.data.site_xpos[self.face_site][2])
            self.apex_z = max(self.apex_z, z)

        if self.cause is None:
            reward += c.alive * self.dt
        reward -= c.action_rate * float(np.sum((a - self.prev_action) ** 2))
        reward -= c.joint_velocity * float(np.sum(w.qvel ** 2))
        self.prev_action = a
        self.t += 1
        self.since_bounce += self.dt

        if self.cause is None:
            ball = w.ball_pos()
            if np.hypot(ball[0], ball[1]) > c.max_reach:
                self.cause = "flew off"
            elif ball[2] < self.r + 0.02:
                self.cause = "dropped"  # on the table, or past its edge
            elif self.since_bounce > c.stall_s:
                self.cause = "stalled"

        terminated = self.cause is not None
        truncated = not terminated and self.t >= self.max_steps
        info = {"bounces": self.bounces}
        if terminated or truncated:
            info["cause"] = self.cause or "timeout"
            info["spawn_dist"] = self.spawn_dist
            if self.flights:
                f = np.array(self.flights)
                info["apex_mean"], info["angle_mean"] = float(f[:, 0].mean()), float(f[:, 1].mean())
            if self.contacts_xy:
                last = np.array(self.contacts_xy[-5:])
                info["home_dist"] = float(np.hypot(*(last - self.home).T).mean())
                info["start_dist"] = float(np.hypot(*(self.contacts_xy[0] - self.home)))
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self._renderer is None:
            wdt, hgt = self._render_size or (640, 480)
            self._renderer = mujoco.Renderer(self.world.model, height=hgt, width=wdt)
        self._renderer.update_scene(self.world.data, camera=self.camera())
        return self._renderer.render()

    def camera(self) -> mujoco.MjvCamera:
        """Behind and above the arm, wide enough to see the whole footprint."""
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = (0.18, 0.0, 0.15)
        cam.distance, cam.azimuth, cam.elevation = 1.25, 160.0, -28.0
        return cam

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # -- internals ------------------------------------------------------------

    def _spawn_point(self) -> np.ndarray:
        """Where this episode's ball is dropped from."""
        c, rng = self.cfg, self.np_random
        z = self.face0[2] + self.r + rng.uniform(*c.drop_height)
        reach = c.spawn_radius * self.spawn_scale
        # Only where the arm can get to before this ball lands. Most of the
        # outer horseshoe needs 0.4-0.5 s against a 0.23-0.35 s fall, and the
        # first v3 run spent 18M steps on a curriculum stage it could not pass.
        fall = np.sqrt(2.0 * (z - self.r - self.face0[2]) / GRAVITY)
        catchable = self.reach_time <= fall if self.reach_time is not None else None
        if (self.footprint is None or reach < FOOTPRINT_STEP or not catchable.any()
                or rng.random() < c.centre_fraction):
            off = rng.uniform(-c.drop_offset, c.drop_offset, 2)
            return np.array([self.home[0] + off[0], self.home[1] + off[1], z])
        dist = np.hypot(*(self.footprint - self.home).T)
        near = self.footprint[(dist <= reach) & catchable]
        inner = c.spawn_radius * self.spawn_scale_prev
        ring = self.footprint[(dist > inner) & (dist <= reach) & catchable]
        if not len(near):
            near = self.footprint[catchable]
        if len(ring) and inner < reach and rng.random() < c.frontier_fraction:
            near = ring
        xy = near[rng.integers(len(near))] + rng.uniform(-0.5, 0.5, 2) * FOOTPRINT_STEP
        return np.array([xy[0], xy[1], z])

    def _randomize_world(self) -> None:
        """Draw this episode's physics and control rate. A no-op without `rand`."""
        R, rng, w = self.cfg.rand, self.np_random, self.world
        m = w.model
        hz = rng.uniform(*R.control_hz) if R else None
        self.substeps = w.steps_per_command if hz is None else max(1, round(1.0 / (hz * m.opt.timestep)))
        self.dt = self.substeps * m.opt.timestep
        self.max_steps = int(round(self.cfg.episode_s / self.dt))
        self.latency = int(rng.choice(R.ball_latency_steps)) if R else 0
        if R is None:
            return

        k = rng.uniform(*R.ball_mass)
        m.body_mass[w.ball_body] = self._ball_mass * k
        m.body_inertia[w.ball_body] = self._ball_inertia * k
        w.cfg = replace(w.cfg, ball=replace(self._ball_cfg,
                                            mass=self._ball_mass * k,
                                            drag_coefficient=rng.uniform(*R.drag_coefficient)))

        self.restitution = rng.uniform(*R.paddle_restitution)
        self._set_restitution(self.restitution)

        act = w.actuator_ids
        kp = self._kp * rng.uniform(*R.servo_kp, 5)
        m.actuator_gainprm[act, 0] = kp
        m.actuator_biasprm[act, 1] = -kp
        m.actuator_biasprm[act, 2] = -self._kv * rng.uniform(*R.servo_kv, 5)
        m.dof_damping[w.qvel_adr] = self._damping * rng.uniform(*R.joint_damping, 5)

    def _set_restitution(self, e: float) -> None:
        solref = replace(self._paddle_cfg, restitution=float(np.clip(e, 0.5, 0.95))).contact_solref
        for g in self.paddle_geoms:
            self.world.model.geom_solref[g] = solref

    def _obs(self) -> np.ndarray:
        w, R = self.world, self.cfg.rand
        face, normal = face_pose(w.model, w.data)
        # The ball as the policy sees it: possibly a decision late.
        now = (w.ball_pos(), w.ball_vel())
        ball_pos, ball_vel = self._seen_ball if self.latency else now
        self._seen_ball = now
        q, qd = w.qpos - self.q0, w.qvel
        if R:
            n = self.np_random.normal
            ball_pos = ball_pos + n(0.0, R.ball_pos_noise, 3)
            ball_vel = ball_vel + n(0.0, R.ball_vel_noise, 3)
            q = q + n(0.0, R.joint_pos_noise, 5)
            qd = qd + n(0.0, R.joint_vel_noise, 5)
        to_home = face - np.array([self.home[0], self.home[1], self.face0[2]])
        period = np.array([self.dt * 30.0])
        return np.concatenate([q, self.target - self.q0, qd, ball_pos - face, ball_vel, normal,
                               to_home, self.prev_action, period]).astype(np.float32)

    def _contacts(self) -> float:
        """Score this substep's ball contacts; set `cause` on a fault.

        Checked every substep, not every control step: a ball-on-rubber contact
        lasts ~3 ms, a tenth of a control period, and would mostly be missed.
        """
        c, w = self.cfg, self.world
        d = w.data
        if not self.armed and not self.in_contact and self.peak > c.min_rise:
            self.armed = True

        touching, normal = False, None
        for i in range(d.ncon):
            con = d.contact[i]
            pair = {con.geom1, con.geom2}
            if self.ball_geom not in pair:
                continue
            if pair & self.paddle_geoms:
                touching, normal = True, con.frame[:3].copy()
            elif self.table_geom in pair:
                self.cause = "dropped"
                return 0.0
            elif pair & self.arm_geoms:
                self.cause = "hit the arm"
                return 0.0

        reward = 0.0
        if touching and not self.in_contact:
            # A new contact. It scores if the ball came from high enough, and
            # struck the face rather than the rim.
            face, fn = face_pose(w.model, d)
            ball = w.ball_pos()
            off = np.hypot(*(ball[:2] - face[:2]))
            R = c.rand
            if R:  # this hit's own restitution, deader towards the rim
                e = self.restitution + self.np_random.normal(0.0, R.restitution_jitter)
                self._set_restitution(e - R.rim_loss * (off / self._paddle_cfg.blade_radius) ** 2)
            quality = face_hit_quality(normal, fn)
            if self.armed and quality > 0.0:
                reward += c.centre * float(np.exp(-((off / c.centre_sigma) ** 2)))
                reward += self._score_flight(d.time, ball[:2])
                self.contacts_xy.append(ball[:2].copy())
                self.bounces += 1
                self.since_bounce = 0.0
                self.armed = False
        if not touching and self.in_contact:
            # Takeoff: a new flight starts. Its direction is set now.
            if c.rand and c.rand.takeoff_scatter_deg > 0:
                w.set_ball(w.ball_pos(), self._scatter(w.ball_vel(), c.rand.takeoff_scatter_deg))
            self.takeoff = (d.time, float(d.site_xpos[self.face_site][2]),
                            *self._takeoff_angles(w.ball_pos(), w.ball_vel()))
            self.peak = -np.inf
            self.apex_z = -np.inf
        self.in_contact = touching
        return reward

    def _scatter(self, vel, sigma_deg: float) -> np.ndarray:
        """`vel` turned by a random small angle, speed unchanged."""
        speed = float(np.linalg.norm(vel))
        if speed < 1e-9:
            return vel
        axis = np.cross(vel / speed, [np.cos(t := self.np_random.uniform(0, 2 * np.pi)), np.sin(t), 0.0])
        n = np.linalg.norm(axis)
        if n < 1e-9:
            return vel
        k, th = axis / n, np.radians(self.np_random.normal(0.0, sigma_deg))
        # Rodrigues: rotate about k by th.
        return vel * np.cos(th) + np.cross(k, vel) * np.sin(th) + k * (k @ vel) * (1 - np.cos(th))

    def _takeoff_angles(self, pos, vel) -> tuple[float, float]:
        """(degrees off the ideal direction, degrees off vertical) at takeoff.

        The ideal keeps this takeoff's vertical speed and adds just enough
        horizontal speed to carry the ball `homing_gain` of the way to the
        centre in the time it will be in the air.
        """
        if vel[2] <= 0.0:
            return 90.0, 90.0
        flight = 2.0 * vel[2] / GRAVITY
        ideal = np.array([*(self.cfg.homing_gain * (self.home - pos[:2]) / flight), vel[2]])
        cos = float(ideal @ vel / (np.linalg.norm(ideal) * np.linalg.norm(vel)))
        off_ideal = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        off_vertical = float(np.degrees(np.arctan2(np.hypot(vel[0], vel[1]), vel[2])))
        return off_ideal, off_vertical

    def _score_flight(self, now: float, landed_xy) -> float:
        """Reward for the flight that just ended on the paddle at `landed_xy`.

        The first contact of an episode ends the drop, which the policy did not
        launch: it earns the flat `catch` bonus instead.
        """
        c = self.cfg
        if self.takeoff is None:
            return c.catch
        t0, face_z, off_ideal, off_vertical = self.takeoff
        rise = self.apex_z - self.r - face_z
        lo, hi = c.height_band
        gap = max(lo - rise, rise - hi, 0.0)
        height = c.low_credit + (1.0 - c.low_credit) * float(np.exp(-((gap / c.height_sigma) ** 2)))
        straight = 0.5 + 0.5 * float(np.exp(-((off_ideal / c.straight_sigma_deg) ** 2)))
        dist = float(np.hypot(*(np.asarray(landed_xy) - self.home)))
        home = 0.5 + 0.5 * float(np.exp(-((dist / c.home_sigma) ** 2)))
        self.flights.append((rise, off_vertical))
        return c.flight_weight * (now - t0) * height * straight * home


def nominal_pose(world, guard) -> np.ndarray:
    """Joint angles that hold the face level at `NOMINAL_FACE`."""
    from .pose import solve_face_pose_ik

    sid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_SITE, FACE_SITE)
    sol = solve_face_pose_ik(world, NOMINAL_FACE, (0.0, 0.0, 1.0), site=sid, guard=guard, seed=0)
    return sol.qpos.copy()


FOOTPRINT_STEP = 0.03  # metres between footprint grid points
_FOOTPRINTS: dict = {}


def face_up_footprint(world, guard, z: float = NOMINAL_FACE[2], q0=None) -> tuple[np.ndarray, np.ndarray]:
    """Every (x, y) over the table where the arm can hold the face level at `z`,
    and the joint angles that do it (solved starting from `q0`).

    Seen from above this is a horseshoe around the base, 0.17-0.44 m out, open
    in front of the arm where it would have to fold into itself. A grid point
    counts if the IK puts the face within 5 mm of it, within 3 degrees of
    level, with the whole arm clear of the table. Computed once per process
    (~10 s) and cached: it depends only on the arm and paddle geometry.
    """
    from .pose import solve_face_pose_ik

    key = (z, world.cfg.paddle, None if q0 is None else tuple(np.round(q0, 6)))
    if key in _FOOTPRINTS:
        return _FOOTPRINTS[key]
    t = world.cfg.table
    sid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_SITE, FACE_SITE)
    pts, poses = [], []
    for x in np.arange(-t.arm_inset + FOOTPRINT_STEP, 0.50, FOOTPRINT_STEP):
        for y in np.arange(-t.width / 2 + FOOTPRINT_STEP, t.width / 2, FOOTPRINT_STEP):
            if np.hypot(x, y) > 0.46:
                continue  # beyond any reach the arm has
            s = solve_face_pose_ik(world, (x, y, z), (0.0, 0.0, 1.0), site=sid, guard=guard,
                                   q0=q0, seed=0, restarts=3, iters=200)
            if s.position_error < 0.005 and s.normal[2] > np.cos(np.radians(3)) and guard.is_safe(s.qpos):
                pts.append((x, y))
                poses.append(s.qpos.copy())
    _FOOTPRINTS[key] = (np.array(pts), np.array(poses))
    return _FOOTPRINTS[key]
