"""The tagged paddle in the sim: it is the printed part, and a camera reads
each marker exactly where the model says it is."""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

stag = pytest.importorskip("stag")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.arm import Arm  # noqa: E402
from sim.cameras import add_cv_camera, look_at  # noqa: E402
from sim.config import MarkerConfig  # noqa: E402
from sim.paddle import PADDLE_BODY, PADDLE_HOME, TAG_HD, TAG_SITE, add_paddle, tagged_paddle  # noqa: E402

W, H = 1280, 720
K = np.array([[905.0, 0, 640], [0, 905, 360], [0, 0, 1]])
CAM_POS = np.array([0.30, -0.60, 0.40])


@pytest.fixture(scope="module")
def rig():
    cfg = tagged_paddle()
    R = look_at(CAM_POS, [0.25, 0.0, 0.12])
    t = -R @ CAM_POS

    def build(spec):
        spec.visual.quality.offsamples = 8
        add_paddle(spec, cfg)
        add_cv_camera(spec, "cv", K, R, t, (W, H))

    arm = Arm(offscreen=(W, H), modify=build)
    return arm, cfg, R, t, mujoco.Renderer(arm.model, H, W)


def test_tags_match_the_printed_layout():
    lay = json.loads((ROOT / "cad" / "out" / "paddle_tags.json").read_text())
    cfg = tagged_paddle()
    assert cfg.blade_diameter == pytest.approx(lay["blade_dia"] / 1000)
    assert cfg.blade_thickness == pytest.approx(lay["blade_thick"] / 1000)
    for face, tag in zip(("front", "back"), cfg.tags):
        f = lay["faces"][face]
        assert tag.id == f["id"] and tag.edge == pytest.approx(f["edge"] / 1000)
        assert np.allclose(tag.centre, [lay["centre_xy"][0] / 1000, lay["centre_xy"][1] / 1000, f["z"] / 1000])
        assert np.allclose(tag.right, f["right"]) and np.allclose(tag.up, f["up"])
        assert np.allclose(tag.outward, f["outward"])


def test_mass_is_the_two_printed_parts():
    pytest.importorskip("build123d")
    sys.path.insert(0, str(ROOT / "cad"))
    import paddle as cad
    V, F, o = [], [], 0
    for n in ("black", "white"):
        v, f = cad.read_stl(ROOT / "sim" / "models" / "paddle_tags" / f"{n}.stl")
        V.append(v)
        F.append(f + o)
        o += len(v)
    p = cad.mass_properties(np.vstack(V), np.vstack(F), cad.PLA_DENSITY)
    cfg = tagged_paddle()
    assert cfg.plastic_mass == pytest.approx(p["mass_g"] / 1000, rel=1e-4)
    assert np.allclose(cfg.plastic_com, np.array(p["com_mm"]) / 1000, atol=1e-6)
    assert np.allclose(cfg.plastic_inertia, p["inertia_kg_m2"], rtol=1e-4, atol=1e-11)


@pytest.mark.parametrize("roll", [0.0, 1.2, 3.0])
def test_camera_reads_each_tag_where_the_model_says(rig, roll):
    arm, cfg, R, t, renderer = rig
    arm.reset(PADDLE_HOME + [0.0, 0.0, 0.0, 0.0, roll])
    renderer.update_scene(arm.data, "cv")
    img = np.ascontiguousarray(renderer.render()[..., ::-1])
    corners, ids, _ = stag.detectMarkers(img, TAG_HD)
    assert ids is not None and len(ids) == 1
    bid = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_BODY, PADDLE_BODY)
    Rb, pb = arm.data.xmat[bid].reshape(3, 3), arm.data.xpos[bid]
    tag = next(g for g in cfg.tags if g.id == int(np.ravel(ids)[0]))
    pc = (tag.corners() @ Rb.T + pb) @ R.T + t
    uv = pc[:, :2] / pc[:, 2:] * K[[0, 1], [0, 1]] + K[:2, 2]
    # Which face shows is decided by the roll; each corner within half a pixel.
    assert np.linalg.norm(corners[0][0] - uv, axis=1).max() < 0.5
    # And the face it read is the one turned towards the camera.
    outward = Rb @ tag.outward
    assert outward @ (CAM_POS - (Rb @ tag.centre + pb)) > 0


def test_a_site_marks_each_tag(rig):
    arm, cfg, *_ = rig
    arm.reset(PADDLE_HOME)
    bid = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_BODY, PADDLE_BODY)
    Rb, pb = arm.data.xmat[bid].reshape(3, 3), arm.data.xpos[bid]
    for tag in cfg.tags:
        sid = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_SITE, TAG_SITE.format(tag.id))
        assert np.allclose(arm.data.site_xpos[sid], Rb @ tag.centre + pb, atol=1e-9)
        assert np.allclose(arm.data.site_xmat[sid].reshape(3, 3)[:, 2], Rb @ tag.outward, atol=1e-9)


def test_table_markers_do_not_reuse_the_paddle_ids():
    assert not set(MarkerConfig().ids) & {g.id for g in tagged_paddle().tags} | {2, 3} & set(MarkerConfig().ids)


def test_each_colour_is_its_own_surface():
    """The inlay stands proud of the black - 0.08 mm on the bed side, 0.1 mm on
    the back - so the slicer sees each colour as a separate surface."""
    from sim.paddle import read_stl
    v, _ = read_stl(ROOT / "sim" / "models" / "paddle_tags" / "white.stl")
    assert v[:, 2].min() == pytest.approx(-0.08, abs=2e-3) and v[:, 2].max() == pytest.approx(3.1, abs=2e-3)
    b, _ = read_stl(ROOT / "cad" / "out" / "charuco_board_black.stl")
    w, _ = read_stl(ROOT / "cad" / "out" / "charuco_board_white.stl")
    assert b[:, 2].min() == pytest.approx(-0.08, abs=2e-3) and w[:, 2].min() == pytest.approx(0.0, abs=2e-3)
