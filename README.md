# robo-ping-pong

Two SO-101 arms learning to rally on a table, trained in simulation and
transferred to hardware.

Status: **phase 1, step 1** - an SO-101 in MuJoCo that goes where it is told.

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
sim/models/so101/   vendored MJCF and meshes
scripts/goto.py     command one joint-space target
scripts/demo.py     cycle waypoints in the live viewer
scripts/render.py   render waypoints to out/demo.mp4, no window needed
tests/test_arm.py   it reaches what it is told, and respects its limits
```
