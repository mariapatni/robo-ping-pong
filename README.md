# robo-ping-pong

Two SO-101 arms learning to rally on a table, trained in simulation and
transferred to hardware.

Status: **phase 1, step 2** - an SO-101 in MuJoCo that goes where it is told,
on a table, with a ball that bounces on it.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

The arm model is vendored in `sim/models/so101/`, taken from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(`robotstudio_so101`). It already carries STS3215-specific actuator gains
(`kp=998.22`, `kv=2.731`, `forcerange=+/-2.94 N.m`) and a `backlash` default
class we will switch on during system identification.

## Run it

Headless, prints where it ended up:

```bash
.venv/bin/python scripts/goto.py --deg 30 -40 50 20 0
```

### See it move

Two ways. **Rendering to video needs nothing special:**

```bash
.venv/bin/python scripts/render.py        # writes out/demo.mp4
.venv/bin/python scripts/render_rally.py  # writes out/rally.mp4 - a launched ball
open out/demo.mp4
```

This is also how we will review training rollouts later, which is the best
defence against a policy quietly gaming the reward.

**The live interactive window** needs `mjpython`, not `python`, because on
macOS the viewer must own the main thread:

```bash
.venv/bin/mjpython scripts/demo.py
.venv/bin/mjpython scripts/goto.py --deg 30 -40 50 20 0 --view
```

If `mjpython` dies with `otool ... exit status 69`, that is not MuJoCo. It
means the Xcode licence has not been accepted. Either:

```bash
sudo xcodebuild -license                              # accept it, or
sudo xcode-select -s /Library/Developer/CommandLineTools   # point at CLT
```

Both need your password, so run them yourself in a terminal. Video rendering
works either way and needs no fix.

Tests:

```bash
.venv/bin/python -m pytest tests -q
```

## The interface

`sim/arm.py` fixes the one contract the whole project depends on: **five joint
position targets, in radians, in a fixed order.**

```
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
```

The gripper is excluded on purpose - the paddle is rigid, so it is held at a
constant position and never commanded. The sim integrates at 200 Hz and we
command at 50 Hz, matching the rate we expect from the real servo bus.

## Layout

```
sim/arm.py          the Arm wrapper - load, command, step, read tip pose
sim/config.py       every physical number: table, ball, launcher ranges
sim/table.py        the table, added to the spec at load time
sim/ball.py         the ball, its quadratic drag, and the scripted feeder
sim/env.py          PingPongEnv = Arm + table + ball, drag applied each step
sim/reach.py        how much of its half the arm can actually reach
sim/models/so101/   vendored MJCF and meshes - never edited, see below
scripts/goto.py     command one joint-space target
scripts/demo.py     cycle waypoints in the live viewer
scripts/render.py   render waypoints to out/demo.mp4, no window needed
scripts/render_rally.py  render a launched ball to out/rally.mp4
scripts/reach.py    print the reachable footprint over the arm's half
tests/test_arm.py   it reaches what it is told, and respects its limits
tests/test_ball.py  it falls, bounces to spec, and is slowed by air
```

The vendored model is never edited. Everything this project adds to the world -
the table, the ball, the finer timestep - goes through the `modify` hook on
`Arm`, which is handed the `MjSpec` before compile.

## The world

Table top is `z = 0`, the floor is at `z = -0.900`. The arm's base is at the
origin, bolted to the middle of its short edge, and owns `x in [0, 0.628]`.
`+x` is the direction of play. All of it lives in `sim/config.py`.

The ball is a 40 mm, 2.7 g sphere with quadratic drag applied by us each step,
`F = -0.5*rho*Cd*A*|v|*v` with `Cd = 0.47`; MuJoCo's own fluid model is the
wrong regime. There is no spin in v1, so no Magnus force.

Two numbers worth knowing before designing a task around this:

- **The arm reaches about 67% of its half**, out to `x = 0.48` of `0.628` m.
  Run `scripts/reach.py` to see the footprint.
- **The environment integrates at 2 kHz**, not the model's 200 Hz. A 2.7 g ball
  on a stiff contact needs it; at 200 Hz the solver injects energy and the ball
  leaves the room.
