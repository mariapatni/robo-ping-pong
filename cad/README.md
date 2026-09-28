# The paddle

A 3D-printable ping-pong paddle that bolts to an SO-101's gripper, modelled
parametrically so the numbers can be argued with rather than reverse-engineered
out of a mesh.

|  |  |
|---|---|
| tool | [build123d](https://build123d.readthedocs.io) 0.13 (Python, scriptable, no GUI, no licence) |
| source | [`paddle.py`](paddle.py) - every dimension is a named constant at the top |
| blade | 110 mm diameter, 2.4 mm thick |
| mass | **30.8 g printed** in PLA (24.8 cm3), **+13.9 g** of rubber = **44.7 g** on the arm |
| inertia about the wrist_roll axis | 3.86e-5 kg m2 |
| print | face-down, flat on the bed, **no supports** |
| fasteners | 3x M2 x 16 self-tapping, into holes the jaw already has |

![the paddle fitted](out/paddle_fitted.png)


## Regenerate it

    uv pip install --python .venv/bin/python -r requirements-cad.txt
    .venv/bin/python cad/paddle.py

which rewrites, from the one script:

    cad/out/paddle.step            the solid, for FreeCAD / Fusion / SolidWorks
    cad/out/paddle.stl             3558 triangles, for the slicer
    sim/models/paddle/paddle.stl   942 triangles, for MuJoCo's visual geom
    cad/out/paddle_props.json      volume, mass, centre of mass, inertia

`sim/paddle.py` does **not** import build123d. Its `PaddleConfig` is a hand-copy
of that JSON, and `tests/test_paddle.py::test_config_matches_cad` fails if the
CAD is re-run and the copy is not updated. So: change a constant, re-run the
script, copy the four numbers it prints into `PaddleConfig`, run the tests.

    .venv/bin/python scripts/render_paddle.py        # rewrites the image above


## Why 110 mm

A regulation blade is about 150 mm across. Three things push back on that, and
only two of them turn out to matter.

**Inertia about the wrist: not the constraint.** A 110 mm blade adds 3.86e-5
kg m2 about the wrist_roll axis. The STS3215's own reflected armature in the
vendored model is `armature="0.028"`, so the paddle is **0.14%** of what that
joint is already accelerating. At 150 mm it would be 1.30e-4, still only 0.41%.
Blade size is essentially free in wrist inertia, and any argument for a small
blade that leans on inertia is wrong.

**Mass and gravity torque: the constraint that is real.** Mass goes as the
square of the diameter, and it is bought straight out of joint speed, which is
already what limits us to ~1.0-1.5 m/s of paddle speed.

| blade | printed | + rubber | total | gravity torque at full extension (453 mm lever) | of the 2.94 N.m servo limit |
|---|---|---|---|---|---|
| 110 mm | 30.8 g | 13.9 g | **44.7 g** | 0.199 N.m | 6.8% |
| 150 mm | 55.1 g | 27.2 g | 82.3 g | 0.355 N.m | 12.1% |

44.7 g is 8.9% of the SO-101's ~500 g rating. 82 g is 16.5%, for a machine whose
binding problem is already that it is not fast enough.

**Table clearance: the argument that actually decides it.** The blade is a disc,
so when it is held vertical - which is how you block and drive - its rim reaches
one radius below its centre. At 55 mm the face centre can sit anywhere from 57 mm
up, which covers the strike band `sim/reach.py` works in (30-220 mm). At 75 mm it
cannot: a regulation blade held vertical with its centre at 120 mm has its rim
45 mm above the table, and anywhere in the lower half of the band it is through
the surface.

**What 110 mm buys, against the 15 mm end-effector error.** A clean face hit
needs the contact point at least a ball radius (20 mm) inside the rim. 55 - 20 =
35 mm of usable half-width, against a 15 mm positioning error: **20 mm of
margin**, better than 2x. At 90 mm it would be 25 - 15 = 10 mm and marginal; at
150 mm it would be 40 mm of margin, and the extra 20 mm costs 38 g and the table
clearance above.

**What was traded away:** forgiveness beyond +/-35 mm, and regulation feel. If
the real error turns out to be worse than 15 mm, raise `BLADE_DIA` - the mount,
the mass properties and the sim all follow from it - and pay for it in speed.

One thing the paddle *gives back*: it puts the hitting face 49 mm beyond the old
gripper frame, and `sim/reach.py` measured over the face instead of the gripper
frame goes from **66.6% of the arm's half (max x = 0.480 m)** to **73.4%
(max x = 0.526 m)**.


## How it mounts, and what is and is not confirmed

The paddle replaces the **moving jaw** - that part comes off the arm and does not
go back - but it bolts to the **fixed jaw** (`wrist_roll_follower_so101_v1`),
which is part of the wrist_roll link itself. A paddle hung off the moving jaw
would take every ball impact through the gripper servo's gear train and horn, so
it would be neither rigid nor good for the servo, and `sim/arm.py` already treats
the end effector as rigid.

**What is confirmed.** The fixed jaw's finger carries a row of **five 1.8 mm
through-holes on a 10.01 mm pitch**, drilled across the finger. These are
measured off the vendored upstream mesh
(`sim/models/so101/assets/wrist_roll_follower_so101_v1.stl`, which came from
mujoco_menagerie, which came from TheRobotStudio's CAD) by fitting cylinders to
the triangles whose normals are perpendicular to the hole axis. Residuals: hole
radii 0.85-0.94 mm over the five, pitches 9.889 / 10.163 / 9.840 / 10.152 mm, and
the five centres are collinear to within 0.1 mm. Ø1.8 mm is exactly the pilot an
M2 self-tapping screw wants, which is consistent with how the rest of the arm is
assembled.

**What is not confirmed.** TheRobotStudio does not publish a dimensioned drawing
or a fastener schedule; the repo README only says to have #0 and #1 Phillips
drivers, from which M2/M3 is an inference, not a fact. So the *intended purpose*
of that hole row is unknown - it may be for a gripper pad insert - and nothing
above comes from a primary text source. Every one of those numbers is therefore a
named constant in `paddle.py` (`JAW_HOLE_DIA`, `JAW_HOLE_S`, `JAW_ROW_DIR_XZ`,
`JAW_HOLE_0_XZ`, `JAW_FACE_Y0`, `JAW_FACE_SLOPE`, `MOUNT_HOLES`). **Put calipers
on the real jaw before you print**: check the hole pitch and the finger's width at
the three tip-most holes, correct the constants, re-run. Nothing else has to
change.

The paddle uses the **three holes nearest the tip** and no others, so no part of
the arm has to be drilled or modified. The finger tapers about 4.9 deg per side;
rather than tilt the whole mating face to match - which would put a shallow
overhang under the part - the face stays flat and perpendicular to the screws and
three bosses of stepped height (1.04 / 1.88 / 2.74 mm) take up the taper.

It mounts on the arm's **-y side**. That is not a free choice: the wrist camera
and its mount occupy 60 mm of the +y side. Mounting on +y needs a mirrored print,
and `PaddleConfig.frame` rejects it rather than silently building a model that
intersects the camera.


## Printing it

**Orientation: hitting face down, flat on the bed. Nothing else.** Everything
else on the part - neck, pad, bosses - grows upwards from that face, so there is
not a single overhang and no supports are needed.

| setting | value | why |
|---|---|---|
| material | PLA | PETG if you keep cracking the neck; it is ~4% heavier |
| nozzle / layer height | 0.4 mm / 0.2 mm | 0.15 mm if you want a nicer glue face |
| walls (perimeters) | **4** | 1.6 mm of the 2.4 mm blade is wall; this is what makes a 2.4 mm disc stiff |
| top / bottom layers | 4 / 4 | at 0.2 mm the blade is 12 layers, so it comes out effectively solid |
| infill | 40% gyroid | only the neck and bosses have any interior at all |
| brim / raft | none | a 110 mm disc on the bed needs no help |
| supports | **none** | |

Print it on a smooth bed (PEI or glass). The first layer is the surface you are
going to glue rubber to.

Expect about 25 cm3, ~10 m of 1.75 mm filament, and 30.8 g.

### The rubber

The face is dead flat, not recessed. A recessed pocket would have to bridge a
100 mm span in mid-air on the first layer, which is the one thing this
orientation cannot do; a flat face is also what contact cement actually wants.
Instead there is an **annular groove, 1.0 mm wide and 0.6 mm deep, on a Ø103 mm
circle**, which is the trim line and a glue relief. Cut the sheet to Ø103 and the
groove tells you when you have it centred.

Use a thin sheet - a pips-out or short-pip rubber around 1.1-1.5 mm, no thick
sponge. The model assumes 1.2 mm at 1.45 g/cm3. This is not cosmetic: it is the
difference between a ball-off-PLA restitution of ~0.60 and ~0.85, i.e. most of
the return speed.

If you would rather have a real pocket, set `RUBBER_RECESS = RUBBER_THICK` in
`paddle.py`. It will need supports.

### Hardware

- **3 x M2 x 16 mm self-tapping (Phillips #0)**, driven from the outside of the
  pad straight into the jaw's existing 1.8 mm holes. Ø1.8 is the correct pilot
  for M2 in PLA; do not drill them out.
- Machine screws plus M2 nuts also work if you would rather not cut threads in
  the jaw - the pad's holes are Ø2.3 clearance.
- Tighten in the order middle, tip, wrist, and stop when the bosses are seated.
  You are clamping into printed plastic.


## Opening and editing it, from scratch

If all you want is to **print it**: `cad/out/paddle.stl` opens in any slicer
(PrusaSlicer, Bambu Studio, Cura, OrcaSlicer). Nothing else is needed.

If you want to **look at or modify the solid** without writing Python, open
`cad/out/paddle.step` in [FreeCAD](https://www.freecad.org) (free, macOS build,
no licence). STEP is a real solid, not a mesh, so it can be edited there.

If you want to **change the parameters**, which is the intended route:

1. `uv pip install --python .venv/bin/python -r requirements-cad.txt` - about
   400 MB, mostly OpenCascade. No GUI and no licence server.
2. Edit the constants at the top of `cad/paddle.py`.
3. `.venv/bin/python cad/paddle.py`.
4. Copy the mass, centre of mass and inertia it prints into
   `sim.paddle.PaddleConfig`, and re-run `pytest tests/test_paddle.py`.

For a live 3D preview while editing, `uv pip install ocp_vscode` plus the
"OCP CAD Viewer" VS Code extension, then `show(part)`. Optional - the script does
not import it.
