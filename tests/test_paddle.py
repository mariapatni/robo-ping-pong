"""The paddle is on the arm, it is rigid, it weighs what the CAD says, and the
ball comes off it as hard as we asked.

Same shape as tests/test_ball.py: build the world, run the physics, check the
number that came back. Two of these are guards against a specific way this can
rot. `test_config_matches_cad` fails if cad/paddle.py is re-run and its numbers
are not carried into PaddleConfig, which is the only link between the two and is
otherwise silent. `test_the_gripper_is_gone` fails if any of the gripper the
paddle replaced is left in the model, where it would still weigh 87 g and still
collide in every rollout.
"""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.arm import ARM_JOINTS, GRIPPER_JOINT, Arm  # noqa: E402
from sim.ball import BALL_BODY, add_ball  # noqa: E402
from sim.config import BallConfig, TableConfig  # noqa: E402
from sim.paddle import (  # noqa: E402
    BLADE_GEOM,
    FACE_SITE,
    GRIPPER_BODIES,
    HOST_BODY,
    PADDLE_BODY,
    PADDLE_HOME,
    PaddleConfig,
    add_paddle,
    face_pose,
    read_stl,
)
from sim.table import add_table  # noqa: E402

PROPS = Path(__file__).resolve().parents[1] / "cad" / "out" / "paddle_props.json"

# A pose that holds the blade face-up, 143 mm above the table, so a ball can be
# dropped on it. Not exactly level - the wrist cannot make it exactly level - so
# the bounce test works in the face normal, not in z.
FACE_UP = np.radians([0.0, -65.0, 93.0, -24.0, 90.0])


def build(cfg: PaddleConfig | None = None, ball: bool = False) -> Arm:
    cfg = cfg or PaddleConfig()
    table = TableConfig()

    def modify(spec: mujoco.MjSpec) -> None:
        spec.option.timestep = 0.0005  # as EnvConfig does: a 2.7 g contact needs it
        add_table(spec, table)
        add_paddle(spec, cfg)
        if ball:
            add_ball(spec, BallConfig(), pos=(0.9, 0.35, 0.02))

    return Arm(modify=modify)


@pytest.fixture(scope="module")
def arm() -> Arm:
    return build()


def body_id(arm: Arm, name: str) -> int:
    return mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_BODY, name)


# -- it is in the model -------------------------------------------------------


def test_model_compiles_with_the_paddle(arm):
    assert body_id(arm, PADDLE_BODY) >= 0
    assert mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_GEOM, BLADE_GEOM) >= 0
    assert mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_SITE, FACE_SITE) >= 0


def test_the_arm_is_left_with_exactly_the_five_joint_interface(arm):
    """Taking the gripper off removes its joint and actuator, and adding the
    paddle must not add any back, or sim/arm.py's contract - five targets, in
    this order - stops being true. Five is also what the servo bus carries."""
    names = [
        mujoco.mj_id2name(arm.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(arm.model.nu)
    ]
    assert names == list(ARM_JOINTS)
    assert arm.model.nq == arm.model.nv == len(ARM_JOINTS)


def test_it_clears_the_table_at_paddle_home(arm):
    """sim.arm.HOME does NOT clear it - see the note on PADDLE_HOME."""
    arm.reset(PADDLE_HOME)
    mujoco.mj_forward(arm.model, arm.data)
    assert arm.data.ncon == 0


# -- it is rigid --------------------------------------------------------------


def test_rigid_no_joint_of_its_own(arm):
    bid = body_id(arm, PADDLE_BODY)
    assert arm.model.body_jntnum[bid] == 0


def test_rigid_welded_to_the_wrist_roll_link(arm):
    """Same weld group as `gripper` is the model-level statement of "rigid":
    MuJoCo has fused the two into one rigid body."""
    assert arm.model.body_weldid[body_id(arm, PADDLE_BODY)] == arm.model.body_weldid[
        body_id(arm, "gripper")
    ]


def test_wrist_roll_does_turn_it(arm):
    """The other half of the same statement: the paddle is rigid, and wrist_roll
    is the joint that aims it."""
    arm.reset(PADDLE_HOME)
    _, n0 = face_pose(arm.model, arm.data)
    q = PADDLE_HOME.copy()
    q[4] += np.pi / 2
    arm.reset(q)
    _, n1 = face_pose(arm.model, arm.data)
    angle = np.degrees(np.arccos(np.clip(n0 @ n1, -1.0, 1.0)))
    assert 88.0 < angle < 92.0


def test_the_gripper_is_gone(arm):
    """On the hardware the fixed jaw, gripper servo, moving jaw and camera mount
    all come off. None of them may survive in the model: not a joint, not a
    collider, not the 87 g the upstream model hangs on the wrist_roll link."""
    assert mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT) < 0
    for name in GRIPPER_BODIES:
        assert body_id(arm, name) < 0
    assert arm.model.ncam == 0

    host = body_id(arm, HOST_BODY)
    on_host = np.flatnonzero(arm.model.geom_bodyid == host)
    assert not (arm.model.geom_contype[on_host] | arm.model.geom_conaffinity[on_host]).any()
    assert arm.model.body_mass[host] < 0.005


def test_the_end_of_the_arm_got_lighter(arm):
    """Paddle and horn together weigh less than the gripper they replace did
    on its own, before the old paddle was even bolted to it."""
    host = body_id(arm, HOST_BODY)
    assert arm.model.body_subtreemass[host] < 0.087


# -- it weighs what it should -------------------------------------------------


def test_mass_is_in_range(arm):
    cfg = PaddleConfig()
    assert 0.020 < cfg.plastic_mass < 0.060, "the print alone must be well under 60 g"
    assert 0.025 < cfg.mass < 0.060, "print plus rubber is the payload the arm carries"
    assert arm.model.body_mass[body_id(arm, PADDLE_BODY)] == pytest.approx(cfg.mass, rel=1e-9)


def test_it_is_a_small_fraction_of_the_payload(arm):
    """500 g is the SO-101's rating and end-effector mass is bought straight out
    of joint speed, which is already the binding constraint. Measured as
    everything wrist_roll turns: with the gripper still fitted that was 156 g."""
    host = body_id(arm, HOST_BODY)
    assert arm.model.body_subtreemass[host] < 0.12 * 0.500


def test_inertia_is_positive_definite_and_disc_like():
    cfg = PaddleConfig()
    w = np.linalg.eigvalsh(cfg.inertia)
    assert (w > 0).all()
    # A flat disc's largest principal moment is about its normal and is about the
    # sum of the other two. 25% either way covers the hub and the gusset.
    assert w[2] == pytest.approx(w[0] + w[1], rel=0.25)


def test_config_matches_cad():
    """PaddleConfig is a hand-copy of cad/out/paddle_props.json. This is the only
    thing that notices when the CAD is changed and the copy is not."""
    props = json.loads(PROPS.read_text())
    cfg = PaddleConfig()
    assert cfg.plastic_mass == pytest.approx(props["mass_g"] * 1e-3, rel=1e-3)
    assert cfg.rubber_mass == pytest.approx(props["rubber_mass_g"] * 1e-3, rel=1e-3)
    assert cfg.blade_diameter == pytest.approx(props["blade_dia_mm"] * 1e-3, rel=1e-6)
    assert cfg.blade_thickness == pytest.approx(props["blade_thick_mm"] * 1e-3, rel=1e-6)
    assert cfg.blade_centre_x == pytest.approx(props["blade_centre_x_mm"] * 1e-3, rel=1e-4)
    assert np.allclose(cfg.plastic_com, np.array(props["com_mm"]) * 1e-3, atol=1e-6)
    assert np.allclose(cfg.plastic_inertia, props["inertia_kg_m2"], atol=1e-9)


def test_the_horn_screws_can_be_driven():
    """The screws go in from the paddle side, so each needs a straight line back
    along its own axis. cad/paddle.py checks this on the solid; this is the same
    check on the committed numbers, so it runs without a CAD stack."""
    props = json.loads(PROPS.read_text())
    driver_r = 2.5
    axis = props["axis_z_mm"]
    for y, z in props["horn_holes_yz_mm"]:
        assert np.hypot(y, z - axis) == pytest.approx(7.0, abs=1e-6), "Ø14 bolt circle"
        assert abs(y) - driver_r > 0.5 * props["rib_width_mm"], "the rib is in the way"
        assert z - driver_r > props["neck_thick_mm"], "the neck web is in the way"


def test_nothing_sweeps_into_the_wrist_bracket():
    """The wrist_roll servo's bracket reaches 4.73 mm past the horn face, outside
    14.1 mm of the roll axis, and wrist_roll swings the paddle round into it.
    MuJoCo cannot see this - it does not collide a link with its parent - so it
    is checked here on the committed numbers (and in cad/paddle.py on the solid).
    """
    props = json.loads(PROPS.read_text())
    depth, bracket_r = 4.73, 14.1
    rim = props["blade_centre_x_mm"] - 0.5 * props["blade_dia_mm"]
    assert rim > depth, "the blade's rim is inside the bracket's reach"
    # The gusset web leaves the hub inside the bracket's depth, so its far
    # corners (on the bed, at the web's edge) must stay inside its radius.
    corner = np.hypot(0.5 * props["neck_width_mm"], props["axis_z_mm"])
    assert corner < bracket_r


def test_it_is_white_like_the_print(arm):
    for suffix in ("_visual", "_rubber"):
        gid = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_GEOM, PADDLE_BODY + suffix)
        assert arm.model.geom_rgba[gid][:3].min() > 0.8


def test_the_sim_mesh_is_small_enough_to_ship():
    verts, faces = read_stl()
    assert faces.shape[0] < 2000, "decimate it: this is a visual geom, not a collider"
    assert verts.shape[0] * 3 > faces.max(), "face indices must point at real vertices"


# -- the face pose can be read ------------------------------------------------


def test_face_pose_is_on_the_blade_axis_and_outward(arm):
    cfg = PaddleConfig()
    arm.reset(PADDLE_HOME)
    pos, normal = face_pose(arm.model, arm.data)

    assert normal == pytest.approx(normal / np.linalg.norm(normal), abs=1e-9)

    gid = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_GEOM, BLADE_GEOM)
    centre = arm.data.geom_xpos[gid]
    axis = arm.data.geom_xmat[gid].reshape(3, 3)[:, 2]

    # The face is on the cylinder's axis, half a blade+rubber ahead of its centre.
    offset = pos - centre
    assert np.linalg.norm(offset - (offset @ axis) * axis) < 1e-9
    half = 0.5 * (cfg.blade_thickness + cfg.rubber_thickness)
    assert offset @ axis == pytest.approx(-half, abs=1e-9)
    # ...and the normal points out of that face, away from the arm.
    assert normal @ axis == pytest.approx(-1.0, abs=1e-9)


def test_the_face_is_ahead_of_the_old_gripper_frame(arm):
    """The paddle extends the arm. Worth asserting, because sim/reach.py's answer
    changes because of it."""
    arm.reset(PADDLE_HOME)
    pos, _ = face_pose(arm.model, arm.data)
    assert 0.03 < np.linalg.norm(pos - arm.tip_pos()) < 0.08


# -- and the ball comes off it --------------------------------------------------


def test_the_ball_bounces_off_the_rubber_at_the_configured_restitution():
    """Drop a ball on the face and read the restitution off the velocity either
    side of the contact, in the face normal - the paddle cannot be made exactly
    level by the wrist, so z would be the wrong axis to measure in.

    This is the number that matters most in the whole module: bare PLA is ~0.60
    and rubber is ~0.85, and that difference is most of the ball's return speed.
    """
    arm = build(ball=True)
    ball_jid = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_JOINT, BALL_BODY)
    qadr = int(arm.model.jnt_qposadr[ball_jid])
    dadr = int(arm.model.jnt_dofadr[ball_jid])

    arm.reset(FACE_UP)
    arm.goto(FACE_UP, timeout=1.0)
    pos, normal = face_pose(arm.model, arm.data)

    drop = 0.12
    arm.data.qpos[qadr : qadr + 3] = pos + normal * (drop + BallConfig().radius)
    arm.data.qpos[qadr + 3 : qadr + 7] = (1.0, 0.0, 0.0, 0.0)
    arm.data.qvel[dadr : dadr + 6] = 0.0
    mujoco.mj_forward(arm.model, arm.data)

    blade = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_GEOM, BLADE_GEOM)
    before = after = None
    previous = 0.0
    for _ in range(int(1.2 / arm.model.opt.timestep)):
        v = float(arm.data.qvel[dadr : dadr + 3] @ normal)
        touching = any(
            blade in (arm.data.contact[i].geom1, arm.data.contact[i].geom2)
            for i in range(arm.data.ncon)
        )
        if touching and before is None:
            before = previous  # the last approach speed before the spring engaged
        elif before is not None and not touching:
            after = v  # the first free speed after it let go
            break
        previous = v
        mujoco.mj_step(arm.model, arm.data)

    assert before is not None and after is not None, "the ball never hit the blade"
    e = -after / before
    assert 0.75 < e < 0.95, f"restitution off the paddle was {e:.3f}"
