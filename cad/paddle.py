"""The paddle end effector, as a parametric build123d model.

Run it to regenerate every artefact the rest of the project consumes:

    .venv/bin/python cad/paddle.py

    cad/out/paddle.step            the editable solid, for a real CAD package
    cad/out/paddle.stl             fine mesh, for slicing and printing
    sim/models/paddle/paddle.stl   coarse mesh, for MuJoCo's visual geom
    cad/out/paddle_props.json      volume, mass, centre of mass, inertia

Nothing in sim/ imports build123d: the two meshes and the JSON are committed, so
the simulator and its tests run without a CAD stack installed. `tests/
test_paddle.py` checks the committed JSON against sim.paddle.PaddleConfig, which
is what catches a change made here and not carried across.


WHAT IT BOLTS TO, AND WHY THERE
-------------------------------

The SO-101 gripper has two jaws. `moving_jaw_so101_v1` hangs off the gripper
servo's output; `wrist_roll_follower_so101_v1` is the fixed jaw, which is part of
the wrist_roll link itself. The paddle replaces the moving jaw - that one comes
off and does not go back - but it is bolted to the FIXED jaw, because a paddle on
the gripper servo's output would take every ball impact through one plastic gear
train and a horn, which is neither rigid nor kind to the servo. Bolted to the
fixed jaw the paddle is rigid relative to the arm and wrist_roll turns it, which
is exactly the contract sim/arm.py already assumes.

The fixed jaw's finger carries a row of five 1.8 mm holes on a 10.01 mm pitch,
drilled straight through it. They are measured off the vendored upstream mesh
(see cad/README.md for the method and the residuals) - not from a dimensioned
drawing, which TheRobotStudio does not publish - so every one of those numbers is
a named constant below and can be corrected against the real part with calipers.
The paddle uses the three holes nearest the tip and no others, so nothing on the
arm has to be drilled or modified.

The finger tapers, ~5.2 deg per side. Rather than tilt the whole mounting face to
match - which would put a shallow overhang on the underside of the part and cost
us support-free printing - the mating face stays flat and perpendicular to the
screws, and three bosses of stepped height take up the taper. They grow upwards
in the print orientation, so they cost nothing.


PRINT ORIENTATION, AND THE ONE THING IT COSTS
---------------------------------------------

The part prints face-down: the hitting face is the first layer, flat on the bed,
and everything else - neck, pad, bosses - grows upwards. No supports anywhere.

That rules out a recessed pocket for the rubber. A pocket on the bed-facing side
would leave a ~100 mm disc of floor bridging over nothing. So the rubber face is
dead flat, which is what contact cement actually wants, and the rubber is located
instead by a shallow annular groove that is both a trim line and a glue relief;
the groove's 1 mm roof is a bridge a slicer will not notice. `RUBBER_RECESS`
exists for anyone who would rather have the true pocket and pay for supports.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from build123d import (
    Circle,
    Part,
    Pos,
    Rectangle,
    RectangleRounded,
    chamfer,
    export_step,
    export_stl,
    extrude,
)

# =============================================================================
# Parameters. Millimetres and degrees throughout; sim/paddle.py converts.
# =============================================================================

# -- the fixed jaw, measured off sim/models/so101/assets/wrist_roll_follower_so101_v1.stl.
#    All of it is expressed in the `gripper` body frame of so101.xml, whose z
#    axis is the wrist_roll axis and whose -z points down the jaw.
JAW_HOLE_DIA = 1.8  # +/- 0.05 over the five holes
JAW_HOLE_0_XZ = (-27.47, -49.84)  # the hole nearest the wrist
JAW_ROW_DIR_XZ = (0.3446, -0.9387)  # unit, pointing down the finger
JAW_HOLE_S = (0.0, 9.889, 20.051, 29.891, 40.043)  # along the row; pitch 10.01 +/- 0.09
JAW_FACE_Y0 = -9.464  # body y of the -y side face at s = 0
JAW_FACE_SLOPE = 0.085  # d(face y)/ds, i.e. the 4.9 deg taper of that face
MOUNT_HOLES = (2, 3, 4)  # which of the five we use: the three nearest the tip
MOUNT_REF_HOLE = 3  # the paddle frame's origin sits on this one

# Which side of the jaw the paddle goes on. -1 is body -y, and it is not a free
# choice: the wrist camera and its mount stick 60 mm out the +y side.
MOUNT_SIDE = -1

# -- the blade. 110 mm is a deliberate 73% of a regulation blade; see cad/README.md
#    for the inertia, torque, mass and table-clearance numbers behind it.
BLADE_DIA = 110.0
BLADE_THICK = 2.4
BLADE_RIM_CHAMFER = 0.8  # on the back face only; the front is the print bed

# -- the rubber sheet. Not modelled as a solid: it is bought, not printed.
RUBBER_THICK = 1.2  # a thin pips-out sheet, no sponge
RUBBER_RECESS = 0.0  # set to RUBBER_THICK for a true pocket - needs supports
GROOVE_INSET = 3.5  # from the rim, so the rubber is trimmed 3.5 mm inside it
GROOVE_WIDTH = 1.0
GROOVE_DEPTH = 0.6

# -- the neck and the mounting pad
PAD_THICK = 3.2
PAD_WIDTH = 18.0
PAD_CORNER_R = 4.0
PAD_BEHIND_FIRST_HOLE = 8.0  # material behind the wrist-most screw
PAD_INTO_BLADE = 6.0  # how far the neck runs under the blade, for the fillet
BOSS_DIA = 6.0  # the jaw's side face is only ~11 mm wide there
SCREW_CLEAR_DIA = 2.3  # M2 clearance; the screw self-taps into the 1.8 mm jaw hole

# The bed plane, given as a body-y coordinate. Everything else is measured from
# it. It sets the standoff between the hitting face and the jaw's centre plane,
# and so how far the blade is offset from the wrist_roll axis.
BED_Y = -12.0

# -- material
PLA_DENSITY = 1.24e-3  # g/mm^3
RUBBER_DENSITY = 1.45e-3  # g/mm^3, thin pips-out sheet

# -- mesh export
STL_TOL_FINE = 0.02
STL_ANG_FINE = 0.15
STL_TOL_SIM = 0.30
STL_ANG_SIM = 0.60

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SIM_MESH = HERE.parent / "sim" / "models" / "paddle" / "paddle.stl"


# =============================================================================
# Derived geometry, in the paddle's own frame
# =============================================================================
#   x  along the jaw's hole row, pointing at the blade
#   y  across it
#   z  the print direction: 0 is the hitting face (the bed), +z is towards the jaw
#
# so the mounting screws run along +z and the blade normal is -z.

BLADE_R = 0.5 * BLADE_DIA
S_REF = JAW_HOLE_S[MOUNT_REF_HOLE]


def hole_x(i: int) -> float:
    """Paddle-frame x of jaw hole `i`."""
    return JAW_HOLE_S[i] - S_REF


def jaw_face_z(i: int) -> float:
    """Paddle-frame z of the jaw's side face at hole `i`, i.e. the boss top.

    The face is |JAW_FACE_Y0 + slope*s| off the jaw's centre plane and the bed is
    |BED_Y| off it, both on the MOUNT_SIDE, so the boss has to make up the
    difference - and the difference shrinks down the finger, which is the taper.
    """
    return abs(BED_Y) - abs(JAW_FACE_Y0 + JAW_FACE_SLOPE * JAW_HOLE_S[i])


MOUNT_X = tuple(hole_x(i) for i in MOUNT_HOLES)
BOSS_TOP_Z = tuple(jaw_face_z(i) for i in MOUNT_HOLES)
BOSS_HEIGHT = tuple(z - PAD_THICK for z in BOSS_TOP_Z)

# The blade sits just clear of the jaw's tip. Its centre is pushed out until the
# rim clears the last mounting hole by PAD_INTO_BLADE.
BLADE_CX = MOUNT_X[-1] + PAD_INTO_BLADE + BLADE_R

PAD_X0 = MOUNT_X[0] - PAD_BEHIND_FIRST_HOLE
PAD_X1 = BLADE_CX - BLADE_R + PAD_INTO_BLADE


# =============================================================================
# The solid
# =============================================================================


def build() -> Part:
    """The paddle as one build123d solid, in the paddle frame, in millimetres."""
    if min(BOSS_HEIGHT) <= 0.2:
        raise ValueError(f"BED_Y={BED_Y} leaves boss heights {BOSS_HEIGHT}; move the bed out")

    blade = extrude(Pos(BLADE_CX, 0) * Circle(BLADE_R), amount=BLADE_THICK)

    pad_len = PAD_X1 - PAD_X0
    neck = extrude(
        Pos(0.5 * (PAD_X0 + PAD_X1), 0) * RectangleRounded(pad_len, PAD_WIDTH, PAD_CORNER_R),
        amount=PAD_THICK,
    )
    part = blade + neck

    # A gusset where the neck meets the blade: without it the step from 3.2 mm to
    # 2.4 mm is the hinge the whole blade folds about on a mishit.
    part = part + extrude(
        Pos(0.5 * (PAD_X1 + PAD_X1 + 14.0), 0) * Rectangle(14.0, PAD_WIDTH), amount=PAD_THICK
    ).intersect(extrude(Pos(BLADE_CX, 0) * Circle(BLADE_R), amount=PAD_THICK))

    for x, h in zip(MOUNT_X, BOSS_HEIGHT):
        part = part + extrude(Pos(x, 0, PAD_THICK) * Circle(0.5 * BOSS_DIA), amount=h)

    # Screw holes, drilled clear through pad and boss.
    for x, top in zip(MOUNT_X, BOSS_TOP_Z):
        part = part - extrude(
            Pos(x, 0, -1.0) * Circle(0.5 * SCREW_CLEAR_DIA), amount=top + 2.0
        )

    # The rubber's trim line and glue relief, cut into the hitting face.
    r_out = BLADE_R - GROOVE_INSET
    groove = extrude(
        Pos(BLADE_CX, 0) * (Circle(r_out) - Circle(r_out - GROOVE_WIDTH)), amount=GROOVE_DEPTH
    )
    part = part - groove

    if RUBBER_RECESS > 0.0:
        part = part - extrude(
            Pos(BLADE_CX, 0) * Circle(r_out - GROOVE_WIDTH), amount=RUBBER_RECESS
        )

    # Chamfer the rim on the back face only. The front face is the first layer
    # and a chamfer there would be an overhang.
    try:
        rim = (
            part.edges()
            .filter_by(lambda e: abs(e.center().Z - BLADE_THICK) < 1e-6)
            .sort_by(lambda e: -e.length)[0]
        )
        part = chamfer(rim, length=BLADE_RIM_CHAMFER)
    except Exception as exc:  # pragma: no cover - cosmetic only
        print(f"  (rim chamfer skipped: {exc})")

    return part


# =============================================================================
# Mass properties, straight off the mesh
# =============================================================================


def read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (n, 3) and triangles (m, 3) from a binary STL, de-duplicated."""
    raw = path.read_bytes()
    n = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
    rec = np.frombuffer(raw[84 : 84 + n * 50], dtype=np.uint8).reshape(n, 50)
    tri = rec[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(np.float64)
    flat = tri.reshape(-1, 3)
    verts, inv = np.unique(np.round(flat, 6), axis=0, return_inverse=True)
    return verts, inv.reshape(-1, 3).astype(np.int32)


def mass_properties(verts: np.ndarray, faces: np.ndarray, density: float) -> dict:
    """Volume, centre of mass and the inertia tensor of a closed triangle mesh.

    Signed tetrahedra from the origin: exact for a closed mesh, and it needs no
    CAD kernel, so the committed JSON can be re-derived from the committed STL
    alone. Inertia is returned about the centre of mass.
    """
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    vol6 = det.sum()
    volume = vol6 / 6.0
    com = (det[:, None] * (a + b + c)).sum(0) / (4.0 * vol6)

    # Second-moment ("covariance") matrix of each tetrahedron about the origin,
    #   C_tet = det/120 * ( 2*sum_i p_i p_i^T + sum_{i != j} p_i p_j^T ),
    # summed with sign exactly as the volumes are.
    P = np.stack([a, b, c], axis=1)  # (m, 3 vertices, 3 coords)
    same = np.einsum("mik,mil->mkl", P, P)  # sum_i p_i p_i^T
    total = np.einsum("mk,ml->mkl", P.sum(1), P.sum(1))  # sum_i sum_j p_i p_j^T
    cov = (det[:, None, None] * (same + total) / 120.0).sum(0)

    mass = density * volume
    # Shift the covariance to the centre of mass, then convert to inertia.
    cov = cov - volume * np.outer(com, com)
    inertia = density * (np.trace(cov) * np.eye(3) - cov)
    return {
        "volume_mm3": float(volume),
        "mass_g": float(mass),
        "com_mm": [float(v) for v in com],
        # MuJoCo fullinertia order, in kg*m^2, about the centre of mass.
        "inertia_kg_m2": [
            float(inertia[0, 0] * 1e-9),
            float(inertia[1, 1] * 1e-9),
            float(inertia[2, 2] * 1e-9),
            float(inertia[0, 1] * 1e-9),
            float(inertia[0, 2] * 1e-9),
            float(inertia[1, 2] * 1e-9),
        ],
    }


# =============================================================================
# main
# =============================================================================


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    SIM_MESH.parent.mkdir(parents=True, exist_ok=True)

    # One fresh solid per export: OCC caches a shape's triangulation, so a second
    # export off the same object silently reuses the first one's tolerance and the
    # "coarse" mesh comes out identical to the fine one.
    export_step(build(), str(OUT / "paddle.step"))
    export_stl(
        build(), str(OUT / "paddle.stl"), tolerance=STL_TOL_FINE, angular_tolerance=STL_ANG_FINE
    )
    export_stl(build(), str(SIM_MESH), tolerance=STL_TOL_SIM, angular_tolerance=STL_ANG_SIM)

    verts, faces = read_stl(OUT / "paddle.stl")
    props = mass_properties(verts, faces, PLA_DENSITY)

    _, sim_faces = read_stl(SIM_MESH)
    rubber_area = math.pi * (BLADE_R - GROOVE_INSET - GROOVE_WIDTH) ** 2
    rubber_mass = RUBBER_DENSITY * rubber_area * RUBBER_THICK

    props.update(
        {
            "blade_dia_mm": BLADE_DIA,
            "blade_thick_mm": BLADE_THICK,
            "blade_centre_x_mm": BLADE_CX,
            "rubber_thick_mm": RUBBER_THICK,
            "rubber_mass_g": float(rubber_mass),
            "mount_hole_x_mm": list(MOUNT_X),
            "boss_top_z_mm": list(BOSS_TOP_Z),
            "bed_y_mm": BED_Y,
            "pad_thick_mm": PAD_THICK,
            "pad_x_mm": [PAD_X0, PAD_X1],
            "pad_width_mm": PAD_WIDTH,
            "fine_triangles": int(faces.shape[0]),
            "sim_triangles": int(sim_faces.shape[0]),
        }
    )
    (OUT / "paddle_props.json").write_text(json.dumps(props, indent=2) + "\n")

    print(f"blade            {BLADE_DIA:.1f} mm dia x {BLADE_THICK:.1f} mm, centre x = {BLADE_CX:.1f}")
    print(f"mount holes x    {np.round(MOUNT_X, 2)}  boss heights {np.round(BOSS_HEIGHT, 2)}")
    print(f"volume           {props['volume_mm3']:.0f} mm^3")
    print(f"PLA mass         {props['mass_g']:.1f} g   (+ {rubber_mass:.1f} g of rubber)")
    print(f"centre of mass   {np.round(props['com_mm'], 2)} mm (paddle frame)")
    print(f"inertia (kg m^2) {np.array(props['inertia_kg_m2'][:3])}")
    print(f"triangles        {faces.shape[0]} fine, {sim_faces.shape[0]} for the sim")
    print(f"wrote {OUT/'paddle.step'}, {OUT/'paddle.stl'}, {SIM_MESH}, {OUT/'paddle_props.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
