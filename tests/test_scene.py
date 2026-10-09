"""The dressed scene: wood, corner markers, net, and the two fixed cameras.

Two kinds of test here. Most are the usual "the thing is where the config says
it is". The interesting ones are the two that pin down a decision:

  * the net height, by measuring where the launcher's feeds actually cross the
    centre line - which is why the net is 60 mm and why collision is off;
  * the camera model, by checking that the pinhole `sim.cameras` does arithmetic
    with lands on the same pixel MuJoCo's renderer draws the ball at. The camera
    study is only worth reading if those two agree.
"""

import sys
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.ball import BALL_GEOM  # noqa: E402
from sim.cameras import (  # noqa: E402
    CAM_A,
    CAM_B,
    camera_pose,
    coverage,
    parallax_deg,
    rig_poses,
    triangulation_sigma,
    visible_fraction,
)
from sim.config import (  # noqa: E402
    CameraConfig,
    EnvConfig,
    MarkerConfig,
    NetConfig,
    SurfaceConfig,
    TableConfig,
)
from sim.env import PingPongEnv  # noqa: E402
from sim.table import MARKER_GEOMS, NET_GEOM, TABLE_GEOM, corner_positions, wood_texture  # noqa: E402


@pytest.fixture(scope="module")
def env():
    return PingPongEnv()


def geom(env, name):
    gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert gid >= 0, f"geom {name!r} missing from the scene"
    return gid


# -- the wooden surface -------------------------------------------------------


def test_wood_texture_is_deterministic_and_varied():
    cfg = SurfaceConfig()
    a, b = wood_texture(cfg), wood_texture(cfg)
    assert a.shape == (cfg.texture_px, cfg.texture_px, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b), "same config must give the same table on every machine"
    assert wood_texture(replace(cfg, seed=cfg.seed + 1)).tobytes() != a.tobytes()
    # Grain, not a flat fill: real variation, and warmer in red than in blue.
    assert a.std() > 15.0
    assert a[..., 0].mean() > a[..., 2].mean() + 40


def test_wood_stays_between_the_two_named_tones():
    cfg = SurfaceConfig()
    img = wood_texture(cfg).astype(float) / 255.0
    for ch, (lo, hi) in enumerate(zip(cfg.dark_rgb, cfg.light_rgb)):
        assert img[..., ch].min() >= min(lo, hi) - 0.02
        assert img[..., ch].max() <= max(lo, hi) + 0.02


def test_the_surface_is_a_choice_not_a_fixture():
    """Turning the wood off must change the look and nothing else."""
    plain = PingPongEnv(replace(EnvConfig(), surface=SurfaceConfig(wood=False)))
    wood = PingPongEnv()
    assert plain.model.ntex < wood.model.ntex
    gp, gw = geom(plain, TABLE_GEOM), geom(wood, TABLE_GEOM)
    assert np.allclose(plain.model.geom_size[gp], wood.model.geom_size[gw])
    assert np.allclose(plain.model.geom_pos[gp], wood.model.geom_pos[gw])
    assert np.allclose(plain.model.geom_solref[gp], wood.model.geom_solref[gw])


def test_table_top_is_still_at_z_zero_after_dressing(env):
    """Regression guard: none of the dressing is allowed to move the surface."""
    t = env.cfg.table
    gid = geom(env, TABLE_GEOM)
    assert env.model.geom_pos[gid][2] + env.model.geom_size[gid][2] == pytest.approx(0.0)
    assert env.model.geom_size[gid][0] == pytest.approx(t.half_length)
    assert env.model.geom_size[gid][1] == pytest.approx(t.half_width)


# -- corner markers -----------------------------------------------------------


def test_four_markers_sit_at_the_four_corners(env):
    t, m = env.cfg.table, env.cfg.markers
    seen = set()
    for name, pos in zip(MARKER_GEOMS, corner_positions(t, m)):
        gid = geom(env, name)
        p = env.model.geom_pos[gid]
        assert np.allclose(p, pos)
        # On the table, clear of both edges by the inset, and never overhanging.
        assert t.x_near + m.radius <= p[0] <= t.x_far - m.radius
        assert abs(p[1]) <= t.half_width - m.radius
        assert p[2] > 0.0, "a marker below the surface is invisible to a camera"
        seen.add((np.sign(p[0] - t.x_net), np.sign(p[1])))
    assert len(seen) == 4, "the four markers must be in four different corners"


def test_each_corner_carries_its_own_fiducial(env):
    """A camera that sees only some corners must still know which ones.

    With STag markers that identity lives in the encoded id, not in the colour,
    so this checks the four geoms carry four different materials. The id itself
    is round-tripped through the real detector in tests/test_fiducials.py.
    """
    mats = [env.model.geom_matid[geom(env, n)] for n in MARKER_GEOMS]
    assert all(m >= 0 for m in mats), "fiducial markers must carry a material"
    assert len(set(mats)) == 4, f"corners share materials: {mats}"


def test_disc_markers_stay_distinct_in_greyscale():
    """The fallback path still has to work: with fiducials off the corners are
    plain discs, and the candidate sensor is monochrome, so they have to differ
    in luma and not only in hue."""
    cfg = EnvConfig(markers=MarkerConfig(fiducial=False))
    world = PingPongEnv(cfg=cfg)
    rgb = np.array([world.model.geom_rgba[geom(world, n)][:3] for n in MARKER_GEOMS])
    grey = rgb @ np.array([0.299, 0.587, 0.114])
    for i in range(4):
        for j in range(i + 1, 4):
            assert np.abs(rgb[i] - rgb[j]).max() > 0.3
            assert abs(grey[i] - grey[j]) > 0.05


def test_markers_do_not_collide(env):
    for name in MARKER_GEOMS:
        gid = geom(env, name)
        assert env.model.geom_contype[gid] == 0
        assert env.model.geom_conaffinity[gid] == 0


def test_a_ball_rolls_over_a_marker(env):
    """The bounds of play must not be a speed bump."""
    t, m = env.cfg.table, env.cfg.markers
    x, y, _ = corner_positions(t, m)[0]
    env.reset()
    env.set_ball((x, y, env.cfg.ball.radius), (0.0, -0.5, 0.0))
    env.step(int(0.3 / env.model.opt.timestep))
    assert env.ball_pos()[2] == pytest.approx(env.cfg.ball.radius, abs=3e-3)


def test_markers_can_be_switched_off():
    off = PingPongEnv(replace(EnvConfig(), markers=MarkerConfig(enabled=False)))
    assert mujoco.mj_name2id(off.model, mujoco.mjtObj.mjOBJ_GEOM, MARKER_GEOMS[0]) < 0


# -- the net ------------------------------------------------------------------


def test_net_stands_on_the_centre_line(env):
    t, n = env.cfg.table, env.cfg.net
    gid = geom(env, NET_GEOM)
    pos, size = env.model.geom_pos[gid], env.model.geom_size[gid]
    assert pos[0] == pytest.approx(t.x_net)
    assert pos[1] == pytest.approx(0.0)
    assert 2 * size[2] == pytest.approx(n.height)
    assert pos[2] - size[2] == pytest.approx(0.0), "the net must reach the surface"
    assert size[1] >= t.half_width, "the net must span the full width"


def test_net_height_leaves_the_arm_room_to_work(env):
    """60 mm is 1.5 ball diameters and still well under the top of the SO-101's
    strike band; a regulation 152 mm net would be above all of it."""
    n = env.cfg.net
    assert 2.5 * env.cfg.ball.radius < n.height < 0.5 * 0.22


def test_net_is_visual_by_default_and_can_be_made_solid(env):
    gid = geom(env, NET_GEOM)
    assert env.model.geom_contype[gid] == 0 and env.model.geom_conaffinity[gid] == 0
    solid = PingPongEnv(replace(EnvConfig(), net=NetConfig(collide=True)))
    sid = geom(solid, NET_GEOM)
    assert solid.model.geom_contype[sid] != 0


def test_a_solid_net_actually_stops_a_ball():
    cfg = replace(EnvConfig(), net=NetConfig(collide=True))
    solid = PingPongEnv(cfg)
    solid.reset()
    # Fired at the net, below its top: it must not come out the other side.
    solid.set_ball((cfg.table.x_net + 0.25, 0.0, 0.03), (-2.0, 0.0, 0.0))
    solid.step(int(0.6 / solid.model.opt.timestep))
    assert solid.ball_pos()[0] > cfg.table.x_net, "the ball went through the net"


def crossing_heights(seeds=range(32)) -> np.ndarray:
    """Height of the ball as it crosses the centre line, per launcher seed.

    This is the measurement the net's `collide=False` default rests on, so it is
    made here rather than asserted from a comment.
    """
    env = PingPongEnv(replace(EnvConfig(), net=NetConfig(enabled=False)))
    mid = env.cfg.table.x_net
    out = []
    for seed in seeds:
        env.reset()
        env.launch(seed)
        prev = env.ball_pos()
        for _ in range(int(2.5 / env.model.opt.timestep)):
            env.step(1)
            p = env.ball_pos()
            if prev[0] > mid >= p[0]:
                f = (prev[0] - mid) / (prev[0] - p[0])
                out.append(prev[2] + f * (p[2] - prev[2]))
                break
            if p[2] < -0.1:
                break
            prev = p
    return np.array(out)


def test_a_regulation_net_would_block_every_feed_we_have():
    """The reason the net is 60 mm AND non-colliding, measured.

    The current LaunchConfig serves the ball low across the centre line. A
    152 mm regulation net would reject all of it, and even the 60 mm net we drew
    would reject a large minority - so switching `collide` on without retuning
    the feeder would silently delete part of the training distribution.
    """
    cfg = EnvConfig()
    r = cfg.ball.radius
    h = crossing_heights()
    assert len(h) >= 30, "feeds should reach the centre line"
    assert float(np.mean(h + r < 0.1525)) > 0.95, "a regulation net would let feeds through"
    blocked = float(np.mean(h < cfg.net.height + r))
    assert 0.2 < blocked < 0.7, f"expected a large minority blocked, got {blocked:.0%}"


# -- the cameras --------------------------------------------------------------


def test_both_cameras_are_in_the_scene(env):
    for name in (CAM_A, CAM_B):
        assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0


def test_camera_optics_come_from_the_lens_we_would_buy(env):
    """MuJoCo must derive the frustum from the OV9281 sensor and the M12 focal
    length, not from a fovy someone picked by eye."""
    cfg = env.cfg.cameras.a
    cid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, CAM_A)
    assert env.model.cam_fovy[cid] == pytest.approx(cfg.fov_v_deg, abs=0.1)
    assert cfg.fov_h_deg == pytest.approx(93.7, abs=0.5)
    assert tuple(env.model.cam_resolution[cid]) == cfg.resolution


def test_each_camera_is_behind_and_above_its_own_arm(env):
    t = env.cfg.table
    a, b = rig_poses(env.cfg.cameras, t)
    assert a.pos[0] < 0.0 and b.pos[0] > t.x_far, "each camera sits behind its own arm"
    assert a.pos[2] > 0.3 and b.pos[2] > 0.3, "and above the table"
    assert a.forward[0] > 0.0 > b.forward[0], "each looks down the table at the other"
    assert a.forward[2] < 0.0 and b.forward[2] < 0.0, "angled downward"
    # Opposite sides of the centre line: that is what makes the pair triangulate.
    assert np.sign(a.pos[1]) == -np.sign(b.pos[1]) != 0


def test_the_pair_sees_the_whole_table(env):
    t = env.cfg.table
    for cam in rig_poses(env.cfg.cameras, t):
        assert coverage(cam, t) > 0.995
        for x in (t.x_near, t.x_far):
            for y in (-t.half_width, t.half_width):
                assert cam.in_frame(np.array([x, y, 0.0])), "a corner fell out of frame"


def test_the_ball_is_big_enough_to_centroid(env):
    t, r = env.cfg.table, env.cfg.ball.radius
    a, _ = rig_poses(env.cfg.cameras, t)
    # Worst case is the far corner of the far half.
    far = np.array([t.x_far, -t.half_width, 0.0])
    assert a.pixels_across(far, 2 * r) > 5.0
    assert a.pixels_across(np.array([0.3, 0.0, 0.1]), 2 * r) > 9.0


@pytest.mark.parametrize("point", [(0.3, 0.0, 0.10), (0.9, -0.2, 0.05), (0.63, 0.25, 0.15)])
def test_the_pinhole_agrees_with_the_renderer(env, point):
    """The camera study's arithmetic and MuJoCo's camera must be one camera.

    Segmentation rendering says exactly which pixels are ball; the pinhole says
    where they should be. If these drift apart, every number in the study is
    describing a camera that does not exist.
    """
    a, _ = rig_poses(env.cfg.cameras, env.cfg.table)
    w, h = env.cfg.cameras.a.resolution
    env.reset()
    env.set_ball(point)
    ball_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, BALL_GEOM)
    with mujoco.Renderer(env.model, h, w) as r:
        r.enable_segmentation_rendering()
        r.update_scene(env.data, camera=CAM_A)
        seg = r.render()[:, :, 0]
    vs, us = np.nonzero(seg == ball_geom)
    assert us.size > 10, "the renderer could not see the ball at all"
    u, v, depth = a.project(np.array(point))
    assert depth > 0
    assert us.mean() == pytest.approx(u, abs=2.0)
    assert vs.mean() == pytest.approx(v, abs=2.0)
    # And the apparent size, which is what the resolution number means.
    assert np.ptp(us) == pytest.approx(a.pixels_across(np.array(point), 2 * env.cfg.ball.radius),
                                     abs=2.0)


def test_occlusion_is_detected(env):
    """A ball in the open is fully visible; a ball tucked behind the arm is not.

    If this ever passes trivially, the occlusion column of the camera study is
    meaningless.
    """
    a, _ = rig_poses(env.cfg.cameras, env.cfg.table)
    r = env.cfg.ball.radius
    env.reset()
    clear = np.array([0.55, -0.25, 0.10])
    env.set_ball(clear)
    assert visible_fraction(env.model, env.data, a, clear, r, env.ball_body) == 1.0

    # Directly behind the arm's elbow, on the line from the camera.
    tip = env.tip_pos()
    hidden = tip + 0.08 * (tip - a.pos) / np.linalg.norm(tip - a.pos)
    env.set_ball(hidden)
    assert visible_fraction(env.model, env.data, a, hidden, r, env.ball_body) < 1.0


# -- two cameras together -----------------------------------------------------


def test_the_pair_is_not_looking_down_one_axis(env):
    """The failure mode the rig exists to avoid: two cameras on the centre line
    staring at each other see the ball along one shared ray and cannot place it
    in depth at all."""
    t = env.cfg.table
    a, b = rig_poses(env.cfg.cameras, t)
    pts = [np.array([x, y, z])
           for x in (0.2, 0.6, 1.0) for y in (-0.25, 0.0, 0.25) for z in (0.04, 0.15)]
    angles = np.array([parallax_deg(a, b, p) for p in pts])
    assert angles.min() > 25.0, f"worst parallax only {angles.min():.1f} deg"
    assert 40.0 < np.median(angles) < 110.0


def test_triangulation_beats_the_degenerate_rig(env):
    """Same lens, same table, cameras on the axis at ball height: the error
    should get dramatically worse, which is what justifies the mount."""
    t = env.cfg.table
    good = rig_poses(env.cfg.cameras, t)
    flat = CameraConfig(height=0.06, back=0.30, pitch_deg=0.0, lateral=0.0)
    bad = (camera_pose(flat, t, "a"), camera_pose(flat, t, "b"))
    pts = [np.array([x, 0.0, 0.10]) for x in (0.25, 0.63, 1.0)]

    for p in pts:
        g = triangulation_sigma(list(good), p)[0]
        d = triangulation_sigma(list(bad), p)[0]
        assert g < 0.006, f"recommended rig should localise to ~mm, got {g * 1000:.1f} mm"
        assert d > 3 * g, "the degenerate rig should be far worse, not slightly worse"


def test_one_camera_cannot_place_the_ball_in_depth(env):
    """Sanity on the metric itself: a single bearing leaves the range free, so
    the worst-axis sigma must be infinite."""
    a, _ = rig_poses(env.cfg.cameras, env.cfg.table)
    worst, best = triangulation_sigma([a], np.array([0.5, 0.0, 0.1]))
    assert not np.isfinite(worst) and np.isfinite(best)
