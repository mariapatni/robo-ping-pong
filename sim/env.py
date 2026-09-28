"""Arm + table + ball: the world a policy will eventually be trained in.

This is an `Arm` with two things added to the world and one thing added to the
step: aerodynamic drag on the ball, which MuJoCo will not do for us. The five
joint targets of `sim.arm` remain the only way to command anything.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .arm import SCENE, Arm
from .ball import BALL_BODY, Launch, add_ball, drag_force, sample_launch
from .config import EnvConfig
from .table import add_table


class PingPongEnv(Arm):
    """The arm on its half of the table, with one ball in play."""

    def __init__(
        self,
        cfg: EnvConfig | None = None,
        scene=SCENE,
        offscreen: tuple[int, int] | None = None,
        **kwargs,
    ):
        self.cfg = cfg or EnvConfig()
        self._ball_ready = False  # Arm.__init__ calls reset() before ids exist
        super().__init__(scene=scene, offscreen=offscreen, modify=self._build, **kwargs)

        self.ball_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, BALL_BODY)
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, BALL_BODY)
        self.ball_qpos_adr = int(self.model.jnt_qposadr[jid])
        self.ball_dof_adr = int(self.model.jnt_dofadr[jid])
        self._ball_ready = True
        self.reset()

    def _build(self, spec: mujoco.MjSpec) -> None:
        spec.option.timestep = self.cfg.timestep
        add_table(spec, self.cfg.table)
        add_ball(spec, self.cfg.ball, pos=self.park_pos)

    # -- ball state -----------------------------------------------------------

    @property
    def park_pos(self) -> tuple[float, float, float]:
        """Where a ball that is not in play sits: resting on the far half, out
        of the arm's way, so a plain reset() is a quiet scene."""
        t = self.cfg.table
        return (0.85 * t.length, 0.35 * t.width, self.cfg.ball.radius)

    def ball_pos(self) -> np.ndarray:
        return self.data.qpos[self.ball_qpos_adr : self.ball_qpos_adr + 3].copy()

    def ball_vel(self) -> np.ndarray:
        return self.data.qvel[self.ball_dof_adr : self.ball_dof_adr + 3].copy()

    def set_ball(self, pos, vel=(0.0, 0.0, 0.0)) -> None:
        """Teleport the ball. Orientation and spin are zeroed: no spin in v1."""
        i, d = self.ball_qpos_adr, self.ball_dof_adr
        self.data.qpos[i : i + 3] = pos
        self.data.qpos[i + 3 : i + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[d : d + 6] = 0.0
        self.data.qvel[d : d + 3] = vel
        mujoco.mj_forward(self.model, self.data)

    # -- stepping -------------------------------------------------------------

    def step(self, n: int = 1) -> None:
        """Step the physics, re-applying drag every substep.

        Drag depends on the current velocity, so it cannot be set once and left:
        `xfrc_applied` persists across steps and would otherwise be a stale
        force. Written as a whole-slice assignment of a pure function's result
        so the MJX version is the same line with `.at[].set()`.
        """
        k = self.cfg.ball.drag_k
        body = self.ball_body
        for _ in range(n):
            self.data.xfrc_applied[body, :3] = drag_force(
                self.data.qvel[self.ball_dof_adr : self.ball_dof_adr + 3], k
            )
            mujoco.mj_step(self.model, self.data)

    def reset(self, q=None) -> None:
        super().reset(q)
        if self._ball_ready:
            self.data.xfrc_applied[:] = 0.0
            self.set_ball(self.park_pos)

    # -- the feeder -----------------------------------------------------------

    def launch(self, seed: int) -> Launch:
        """Put a sampled feed into play. Returns it, so a caller can compare
        the outcome against what was asked for."""
        feed = sample_launch(self.cfg.launch, seed, self.cfg.table.length)
        self.set_ball(feed.pos, feed.vel)
        return feed
