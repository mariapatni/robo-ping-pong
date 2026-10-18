"""The paddle end effector, as a parametric build123d model.

Run it to regenerate every artefact the rest of the project consumes:

    .venv/bin/python cad/paddle.py

    cad/out/paddle.step            the editable solid, for a real CAD package
    cad/out/paddle.stl             fine mesh, for slicing and printing
    sim/models/paddle/paddle.stl   coarse mesh, for MuJoCo's visual geom
    cad/out/paddle_props.json      volume, mass, centre of mass, inertia

and, for printing only, the big paddle - the same hub on a 150 mm blade:

    .venv/bin/python cad/paddle.py --big

    cad/out/paddle_big.step, paddle_big.stl, paddle_big_props.json

The big one does not touch the sim mesh or paddle_props.json: the sim, its
tests and the policies are all built on the 110 mm blade.

Nothing in sim/ imports build123d: the two meshes and the JSON are committed, so
the simulator and its tests run without a CAD stack installed. `tests/
test_paddle.py` checks the committed JSON against sim.paddle.PaddleConfig, which
is what catches a change made here and not carried across.


WHAT IT BOLTS TO, AND WHY THERE
-------------------------------

Straight onto the wrist_roll servo's horn. The whole gripper - the fixed jaw
(`wrist_roll_follower_so101_v1`), the gripper servo inside it, the moving jaw
and the camera mount - comes off the arm, and the paddle's own hub takes the
fixed jaw's place. That sheds 111 g - gripper servo, both jaws, camera mount - from the end
of the arm, and leaves five servos on the bus instead of six.

The hub copies the fixed jaw's horn interface exactly, so the paddle goes on
with the four M3 x 6 mm screws that held the jaw on (the LeRobot SO-101
assembly guide: "attach it to the motor horn on the wrist using 4 M3x6mm
screws"). The interface is measured off the vendored jaw mesh
(sim/models/so101/assets/wrist_roll_follower_so101_v1.stl) by slicing it at the
mating face, and cross-checked against the horn in sts3215_03a_v1.stl:

    four Ø3.2 holes on a Ø14 circle, at 45 deg     horn: four Ø2.5 on Ø14, at 45 deg
    a Ø5.4 centre hole, for the horn hub and screw
    a 1 mm lip, r = 10..12 mm, open on one side     horn disc: Ø20 x 2.1 mm

The screws go in from the paddle side and thread into the horn, so the heads
need a straight line back along the roll axis for a screwdriver. So the short
gusset joining hub to blade is a flat web lying in the blade's own plane,
offset to the hitting-face side of the axis, and it and its stiffening rib pass
BETWEEN the four screw axes rather than across them. `build()` checks it - a Ø5
driver along each screw axis must not touch the part.

The other constraint is the wrist itself. The bracket holding the wrist_roll
servo reaches 4.7 mm past the horn face, outside 14.1 mm of the axis, and
wrist_roll sweeps everything round into it. That is what sets how short the
paddle can be: the blade starts 6 mm from the horn, its centre is 61 mm from
it, and the hub and lip stand on teardrop feet instead of square ones.
`build()` checks that too.


PRINT ORIENTATION
-----------------

Hitting face down, flat on the bed, as before. The blade, the gusset web and
the rib grow upwards from it; the hub is a vertical plate standing on a
teardrop foot - 45 deg flanks into a round top - so it needs no supports
either. Its holes are horizontal, which every slicer prints without help.

There is no recessed pocket for the rubber, for the same reason as before: a
pocket on the bed-facing side would be a 100 mm bridge. The rubber is located
by a shallow annular groove that is both a trim line and a glue relief.
`RUBBER_RECESS` exists for anyone who would rather have the pocket and pay for
supports.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from build123d import (
    Circle,
    Part,
    Plane,
    Polygon,
    Pos,
    Rectangle,
    chamfer,
    export_step,
    export_stl,
    extrude,
)

# =============================================================================
# Parameters. Millimetres and degrees throughout; sim/paddle.py converts.
# =============================================================================

# -- the wrist_roll horn interface, measured off the fixed jaw it replaces. In
#    the `gripper` body frame of so101.xml (the wrist_roll link) the mating face
#    is the plane z = 0, the roll axis is the z axis, and the horn is on +z.
HORN_HOLE_DIA = 3.2  # M3 clearance; the screw threads into the horn's Ø2.5 hole
HORN_HOLE_PCD = 14.0
HORN_HOLE_ANGLES = (45.0, 135.0, 225.0, 315.0)
HORN_CENTRE_DIA = 5.4  # clears the horn's hub and its centre screw
HORN_LIP_R = (10.0, 12.0)  # the Ø20 horn disc sits inside it
HORN_LIP_HEIGHT = 1.0
HORN_LIP_OPEN_X = -3.7  # gripper-frame x: the lip stops here, as the jaw's does

# -- the hub that replaces the jaw. 3.0 mm is the jaw's own plate, which is what
#    makes the original M3 x 6 screws the right length: 3 mm through the plate,
#    the rest into the horn.
HUB_THICK = 3.0
HUB_R = 12.5

# The roll axis sits this far above the hitting face (the print bed). It has to
# be at least HUB_R for the hub to stand on the bed, and it is very nearly where
# the old jaw-mounted paddle had its face (12 mm off the jaw's centre plane).
AXIS_Z = HUB_R

# -- the blade. 110 mm is a deliberate 73% of a regulation blade; see cad/README.md
#    for the inertia, torque, mass and table-clearance numbers behind it.
BLADE_DIA = 110.0
BIG_BLADE_DIA = 150.0  # `--big`: a regulation blade's width, round
BLADE_THICK = 2.4
BLADE_RIM_CHAMFER = 0.8  # on the back face only; the front is the print bed

# -- the wrist bracket. `wrist_roll_pitch_so101_v2` reaches 4.73 mm past the
#    horn face, at 14.1 mm or more from the roll axis. wrist_roll sweeps the
#    paddle through +/-157 deg, so anything of the paddle outside that radius has
#    to stay deeper than that, at every roll angle. `build()` checks it.
WRIST_BRACKET_DEPTH = 4.73
WRIST_BRACKET_R = 14.1
WRIST_CLEARANCE = 1.0

# The blade as close to the horn as the bracket allows: no stick. Its rim
# starts BLADE_GAP past the hub, which puts it 1.3 mm deeper than the bracket.
# That is the shortest lever arm there is - the blade centre is 61 mm from the
# horn face. The jaw-mounted paddle had it at 145.
BLADE_GAP = 3.0
BLADE_CX = HUB_THICK + BLADE_GAP + 0.5 * BLADE_DIA

# -- the rubber sheet. Not modelled as a solid: it is bought, not printed.
RUBBER_THICK = 1.2  # a thin pips-out sheet, no sponge
RUBBER_RECESS = 0.0  # set to RUBBER_THICK for a true pocket - needs supports
GROOVE_INSET = 3.5  # from the rim, so the rubber is trimmed 3.5 mm inside it
GROOVE_WIDTH = 1.0
GROOVE_DEPTH = 0.6

# -- what is left of the neck: a short gusset from hub to blade. A flat web in
#    the blade plane, narrow enough to stay inside WRIST_BRACKET_R where it leaves
#    the hub, plus a rib on its back. Both pass between the four screw axes.
NECK_WIDTH = 12.0
NECK_THICK = 4.0
NECK_INTO_BLADE = 20.0  # how far the web runs under the blade
RIB_WIDTH = 3.0
RIB_TOP = 9.0  # rib height at the hub, from the bed; it tapers into the blade
RIB_INTO_BLADE = 25.0

# A screwdriver has to reach each screw head straight down the screw's axis.
DRIVER_DIA = 5.0

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
#   x  down the roll axis, from the horn face towards the blade
#   y  across the blade
#   z  the print direction: 0 is the hitting face (the bed), +z is the back
#
# In the `gripper` body frame that is x = -z_g, y = -x_g, z = y_g + AXIS_Z: the
# hitting face looks along the body's -y, which is the side the old paddle faced
# too. sim/paddle.py builds the same frame from the same numbers.

BLADE_R = 0.5 * BLADE_DIA


def horn_holes() -> list[tuple[float, float]]:
    """(y, z) of the four horn screws, in the paddle frame."""
    r = 0.5 * HORN_HOLE_PCD
    out = []
    for a in HORN_HOLE_ANGLES:
        xg, yg = r * math.cos(math.radians(a)), r * math.sin(math.radians(a))
        out.append((-xg, yg + AXIS_Z))
    return out


def teardrop(r: float):
    """A circle of radius r on the roll axis, standing on the bed on a 45 deg
    foot. The flanks meet the circle tangentially, so it prints without
    supports; and unlike a square foot its corners stay close to the axis,
    inside WRIST_BRACKET_R."""
    t = r / math.sqrt(2.0)
    ty, tz = t, AXIS_Z - t  # the 45 deg tangent point
    half = ty - tz  # the flank z - tz = y - ty reaches the bed here
    return Pos(0, AXIS_Z) * Circle(r) + Polygon(
        (-half, 0.0), (half, 0.0), (ty, tz), (0.0, AXIS_Z), (-ty, tz), align=None
    )


NECK_X1 = BLADE_CX - BLADE_R + NECK_INTO_BLADE
RIB_X1 = BLADE_CX - BLADE_R + RIB_INTO_BLADE


# =============================================================================
# The solid
# =============================================================================


def build() -> Part:
    """The paddle as one build123d solid, in the paddle frame, in millimetres."""
    blade = extrude(Pos(BLADE_CX, 0) * Circle(BLADE_R), amount=BLADE_THICK)

    # The hub: a round plate on the roll axis, standing on a teardrop foot.
    hub = extrude(Plane.YZ * teardrop(HUB_R), amount=HUB_THICK)

    # The lip that locates the horn disc. Filled down to the bed under the horn,
    # so it has no overhang of its own, and cut open where the jaw's is.
    lip_profile = teardrop(HORN_LIP_R[1]) - Pos(0, AXIS_Z) * Circle(HORN_LIP_R[0])
    keep = Pos(-HORN_LIP_OPEN_X - 50.0, AXIS_Z) * Rectangle(100.0, 100.0)  # y <= -x_open
    lip = extrude(Plane.YZ.offset(-HORN_LIP_HEIGHT) * (lip_profile & keep), amount=HORN_LIP_HEIGHT)

    neck = extrude(
        Pos(0.5 * (HUB_THICK + NECK_X1), 0) * Rectangle(NECK_X1 - HUB_THICK, NECK_WIDTH),
        amount=NECK_THICK,
    )
    rib = extrude(
        Plane.XZ
        * Polygon(
            (HUB_THICK, 0.0),
            (HUB_THICK, RIB_TOP),
            (RIB_X1, BLADE_THICK),
            (RIB_X1, 0.0),
            align=None,
        ),
        amount=0.5 * RIB_WIDTH,
        both=True,
    )
    part = blade + hub + lip + neck + rib

    # Through the hub and lip: the four screws and the centre clearance.
    def bore(y: float, z: float, dia: float):
        return extrude(
            Plane.YZ.offset(-HORN_LIP_HEIGHT - 1.0) * Pos(y, z) * Circle(0.5 * dia),
            amount=HUB_THICK + HORN_LIP_HEIGHT + 2.0,
        )

    for y, z in horn_holes():
        part = part - bore(y, z, HORN_HOLE_DIA)
    part = part - bore(0.0, AXIS_Z, HORN_CENTRE_DIA)

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

    check_driver_access(part)
    check_wrist_clearance(part)
    return part


def check_wrist_clearance(part: Part) -> None:
    """Nothing outside WRIST_BRACKET_R may sit within the bracket's depth.

    Checked as a solid: everything from the horn face to the bracket depth plus
    clearance, minus a cylinder of WRIST_BRACKET_R on the roll axis, must be
    empty - because wrist_roll will carry whatever is left round into it.
    """
    depth = WRIST_BRACKET_DEPTH + WRIST_CLEARANCE
    slab = extrude(
        Plane.YZ.offset(-HORN_LIP_HEIGHT) * Pos(0, AXIS_Z) * Rectangle(400.0, 400.0),
        amount=HORN_LIP_HEIGHT + depth,
    )
    core = extrude(
        Plane.YZ.offset(-HORN_LIP_HEIGHT) * Pos(0, AXIS_Z) * Circle(WRIST_BRACKET_R),
        amount=HORN_LIP_HEIGHT + depth,
    )
    clash = (part & (slab - core)).volume
    if clash > 1e-6:
        raise ValueError(f"{clash:.2f} mm^3 of paddle would sweep into the wrist bracket")


def check_driver_access(part: Part) -> None:
    """Every screw must be reachable straight down its own axis.

    The screws are driven from the paddle side into the horn, so a part that
    fouls a screwdriver cannot be fitted at all - and that failure would only
    show up on the bench, after printing.
    """
    for y, z in horn_holes():
        shaft = extrude(
            Plane.YZ.offset(HUB_THICK + 0.01) * Pos(y, z) * Circle(0.5 * DRIVER_DIA),
            amount=400.0,
        )
        clash = (part & shaft).volume
        if clash > 1e-6:
            raise ValueError(f"the screw at y={y:.2f}, z={z:.2f} cannot be driven: {clash:.2f} mm^3 in the way")


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


def set_blade(dia: float) -> None:
    """Resize the blade, and everything placed off it. The hub, the horn
    interface and the gap to the wrist bracket do not change."""
    global BLADE_DIA, BLADE_R, BLADE_CX, NECK_X1, RIB_X1
    BLADE_DIA = dia
    BLADE_R = 0.5 * dia
    BLADE_CX = HUB_THICK + BLADE_GAP + BLADE_R
    NECK_X1 = BLADE_CX - BLADE_R + NECK_INTO_BLADE
    RIB_X1 = BLADE_CX - BLADE_R + RIB_INTO_BLADE


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the paddle and export it.")
    p.add_argument("--big", action="store_true", help=f"the {BIG_BLADE_DIA:.0f} mm blade, for printing only")
    args = p.parse_args(argv)

    name = "paddle"
    if args.big:
        set_blade(BIG_BLADE_DIA)
        name = "paddle_big"

    OUT.mkdir(parents=True, exist_ok=True)
    SIM_MESH.parent.mkdir(parents=True, exist_ok=True)

    # One fresh solid per export: OCC caches a shape's triangulation, so a second
    # export off the same object silently reuses the first one's tolerance and the
    # "coarse" mesh comes out identical to the fine one.
    export_step(build(), str(OUT / f"{name}.step"))
    export_stl(
        build(), str(OUT / f"{name}.stl"), tolerance=STL_TOL_FINE, angular_tolerance=STL_ANG_FINE
    )
    if not args.big:
        export_stl(build(), str(SIM_MESH), tolerance=STL_TOL_SIM, angular_tolerance=STL_ANG_SIM)

    verts, faces = read_stl(OUT / f"{name}.stl")
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
            "axis_z_mm": AXIS_Z,
            "hub_thick_mm": HUB_THICK,
            "hub_r_mm": HUB_R,
            "horn_holes_yz_mm": [list(h) for h in horn_holes()],
            "blade_gap_mm": BLADE_GAP,
            "neck_x_mm": [HUB_THICK, NECK_X1],
            "neck_width_mm": NECK_WIDTH,
            "neck_thick_mm": NECK_THICK,
            "rib_width_mm": RIB_WIDTH,
            "rib_top_mm": RIB_TOP,
            "fine_triangles": int(faces.shape[0]),
        }
    )
    if not args.big:
        props["sim_triangles"] = int(sim_faces.shape[0])
    (OUT / f"{name}_props.json").write_text(json.dumps(props, indent=2) + "\n")

    print(f"blade            {BLADE_DIA:.1f} mm dia x {BLADE_THICK:.1f} mm, centre x = {BLADE_CX:.1f}")
    print(f"horn screws (yz) {np.round(horn_holes(), 2).tolist()}")
    print(f"volume           {props['volume_mm3']:.0f} mm^3")
    print(f"PLA mass         {props['mass_g']:.1f} g   (+ {rubber_mass:.1f} g of rubber)")
    print(f"centre of mass   {np.round(props['com_mm'], 2)} mm (paddle frame)")
    print(f"inertia (kg m^2) {np.array(props['inertia_kg_m2'])}")
    print(f"triangles        {faces.shape[0]} fine")
    print(f"length           {BLADE_CX + BLADE_R + HORN_LIP_HEIGHT:.1f} mm, horn lip to blade tip")
    wrote = [OUT / f"{name}.step", OUT / f"{name}.stl", OUT / f"{name}_props.json"]
    if not args.big:
        wrote.insert(2, SIM_MESH)
    print("wrote " + ", ".join(str(w) for w in wrote))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
