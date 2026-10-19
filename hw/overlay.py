"""Draw the sim's arm over a webcam frame, from the calibrated camera's viewpoint.

    .venv/bin/python -m hw.overlay                     # live: the arm as it stands now
    .venv/bin/python -m hw.overlay --frame f.jpg --q 0 -0.3 0.9 -0.5 0

Where the yellow sim arm and the real arm disagree, either the camera or the
arm's calibration is off. Writes out/hw/overlay.jpg.
"""

from __future__ import annotations

import argparse

import cv2
import mujoco
import numpy as np

from .vision import ROOT, CameraModel


class Overlay:
    def __init__(self, cam: CameraModel, world):
        self.cam, self.world = cam, world
        m = world.model
        m.vis.global_.fovy = float(np.degrees(2 * np.arctan(cam.height / 2 / cam.f)))
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, cam.width)
        m.vis.global_.offheight = max(m.vis.global_.offheight, cam.height)
        self.r = mujoco.Renderer(m, cam.height, cam.width)
        self.mjcam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()

    def render(self, q) -> tuple[np.ndarray, np.ndarray]:
        """(BGR image, mask of the arm) of the sim at joint angles q, seen by the webcam."""
        w = self.world
        w.reset(np.asarray(q, float))
        w.set_ball((5.0, 5.0, -5.0))
        mujoco.mj_forward(w.model, w.data)
        self.r.update_scene(w.data, self.mjcam, self.opt)
        R = self.cam.R
        for gl in self.r.scene.camera:
            gl.pos[:] = self.cam.position
            gl.forward[:] = R[2]
            gl.up[:] = -R[1]
        rgb = self.r.render()
        self.r.enable_segmentation_rendering()
        self.r.update_scene(w.data, self.mjcam, self.opt)
        for gl in self.r.scene.camera:
            gl.pos[:] = self.cam.position
            gl.forward[:] = R[2]
            gl.up[:] = -R[1]
        seg = self.r.render()
        self.r.disable_segmentation_rendering()
        m = w.model
        arm_bodies = {b for b in range(m.nbody) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
                      in ("upper_arm", "lower_arm", "wrist", "paddle", "shoulder", "gripper", "base")}
        geom_ok = np.array([m.geom_bodyid[g] in arm_bodies for g in range(m.ngeom)])
        ids = seg[..., 0]
        mask = (ids >= 0) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & geom_ok[np.clip(ids, 0, m.ngeom - 1)]
        return rgb[..., ::-1].copy(), mask

    def draw(self, frame: np.ndarray, q, colour=(0, 220, 255), label: str = "") -> np.ndarray:
        _, mask = self.render(q)
        out = frame.copy()
        tint = out.copy()
        tint[mask] = colour
        out = cv2.addWeighted(out, 0.6, tint, 0.4, 0)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, colour, 2)
        if label:
            cv2.putText(out, label, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, label, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frame")
    p.add_argument("--q", nargs=5, type=float)
    args = p.parse_args(argv)
    from .so101 import SimCheck
    cam = CameraModel.load()
    world = SimCheck().world
    if args.frame:
        frame, q = cv2.imread(args.frame), np.array(args.q)
    else:
        from .so101 import SO101
        from .vision import Camera
        c = Camera()
        frame = c.read()[0]
        c.close()
        q = SO101().q()
    cv2.imwrite(str(ROOT / "out" / "hw" / "overlay.jpg"), Overlay(cam, world).draw(frame, q, label=f"q {np.round(q, 2)}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
