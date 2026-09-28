"""Minimal wrapper around the SO-101 MuJoCo model.

The job of this module is to pin down ONE interface that the simulator and the
real arm will both speak: five joint position targets, in radians, in a fixed
order. Everything else in the project is built on top of that contract, so it
is worth keeping small and boring.

The gripper is deliberately not part of the interface. The paddle is rigid, so
the gripper is held at a constant position and never commanded.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "models" / "so101"
SCENE = MODEL_DIR / "scene.xml"

# Order matters and must never change: it is the order of the policy's action
# vector, and the order we will write to the real servo bus.
ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER_JOINT = "gripper"

# A neutral, arm-up pose to start from and return to.
HOME = np.array([0.0, -0.6, 0.9, 0.5, 0.0])

# The model integrates at 200 Hz; we command at 50 Hz, matching the rate we
# expect to achieve on the real servo bus.
CONTROL_HZ = 50


class Arm:
    """An SO-101 in MuJoCo, commanded by joint position targets."""

    def __init__(
        self,
        scene: Path | str = SCENE,
        gripper_hold: float = 0.0,
        offscreen: tuple[int, int] | None = None,
    ):
        if offscreen is None:
            self.model = mujoco.MjModel.from_xml_path(str(scene))
        else:
            # Raise the offscreen framebuffer for video rendering, without
            # editing the vendored model files.
            spec = mujoco.MjSpec.from_file(str(scene))
            spec.visual.global_.offwidth, spec.visual.global_.offheight = offscreen
            self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.gripper_hold = float(gripper_hold)

        self.steps_per_command = max(1, round(1.0 / (CONTROL_HZ * self.model.opt.timestep)))

        def joint_id(name: str) -> int:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise KeyError(f"joint {name!r} not in model")
            return jid

        def actuator_id(name: str) -> int:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise KeyError(f"actuator {name!r} not in model")
            return aid

        self.joint_ids = np.array([joint_id(n) for n in ARM_JOINTS])
        self.actuator_ids = np.array([actuator_id(n) for n in ARM_JOINTS])
        self.qpos_adr = self.model.jnt_qposadr[self.joint_ids]
        self.qvel_adr = self.model.jnt_dofadr[self.joint_ids]

        self.gripper_actuator = actuator_id(GRIPPER_JOINT)
        self.gripper_qpos_adr = self.model.jnt_qposadr[joint_id(GRIPPER_JOINT)]

        self.lower = self.model.jnt_range[self.joint_ids, 0].copy()
        self.upper = self.model.jnt_range[self.joint_ids, 1].copy()

        self.tip_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

        self.reset()

    # -- state ----------------------------------------------------------------

    @property
    def qpos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    @property
    def qvel(self) -> np.ndarray:
        return self.data.qvel[self.qvel_adr].copy()

    def tip_pos(self) -> np.ndarray:
        """Position of the gripper frame in world coordinates. The paddle will
        be rigidly attached here, so this is the thing we ultimately aim."""
        return self.data.site_xpos[self.tip_site].copy()

    def tip_mat(self) -> np.ndarray:
        """Orientation of the gripper frame as a 3x3 rotation matrix."""
        return self.data.site_xmat[self.tip_site].reshape(3, 3).copy()

    # -- control --------------------------------------------------------------

    def clip(self, q) -> np.ndarray:
        return np.clip(np.asarray(q, dtype=float), self.lower, self.upper)

    def command(self, q) -> np.ndarray:
        """Set the joint position targets. Returns the clipped targets actually
        applied, so callers can see when they asked for something unreachable."""
        target = self.clip(q)
        self.data.ctrl[self.actuator_ids] = target
        self.data.ctrl[self.gripper_actuator] = self.gripper_hold
        return target

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def reset(self, q=None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        q = HOME if q is None else q
        target = self.clip(q)
        self.data.qpos[self.qpos_adr] = target
        self.data.qpos[self.gripper_qpos_adr] = self.gripper_hold
        self.command(target)
        mujoco.mj_forward(self.model, self.data)

    def goto(self, q, timeout: float = 4.0, tol: float = 5e-3, vel_tol: float = 1e-2) -> dict:
        """Command a target and run until the arm settles or `timeout` expires.

        Returns a small report: whether it converged, how long it took, and the
        worst per-joint error. Useful as a sanity check now and as the basis of
        the open-loop replay test we will run against the real arm later.
        """
        target = self.command(q)
        deadline = self.data.time + timeout
        while self.data.time < deadline:
            self.step(self.steps_per_command)
            err = np.abs(self.qpos - target)
            if err.max() < tol and np.abs(self.qvel).max() < vel_tol:
                break
        err = np.abs(self.qpos - target)
        return {
            "target": target,
            "reached": self.qpos,
            "max_error_rad": float(err.max()),
            "max_error_deg": float(np.degrees(err.max())),
            "settled": bool(err.max() < tol),
            "tip_pos": self.tip_pos(),
        }
