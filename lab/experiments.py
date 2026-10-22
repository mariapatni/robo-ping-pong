"""The sim experiments the dashboard can start, and what each one is.

Each entry is shown on the dashboard's Live page with its description, and
started from there with a button. Only these commands can be launched - the
dashboard passes an experiment name and whitelisted integer arguments, never
a command line.
"""

from __future__ import annotations

PY = ".venv/bin/python"

EXPERIMENTS = {
    "hover_touch": {
        "title": "Hover-the-ball touch test (sim)",
        "kind": "calibration",
        "description": (
            "A fixed orange ball hangs somewhere over the table. The camera finds it and puts its centre in "
            "the world frame; the arm brings the paddle up face-flat, centre on the vertical through the ball, "
            "and climbs 1 mm at a time until it touches. Where the face is at first touch is the truth: "
            "sideways error = face centre minus ball centre, height error = where the camera said the touch "
            "would come minus where it came. Pass: within 10 mm on every axis at every spot."),
        "args": {"n": 20, "seed": 0},
        "command": lambda a: [PY, "-m", "hw.hover_touch", "--sim", "--n", str(a["n"]), "--seed", str(a["seed"])],
    },
    "handeye": {
        "title": "Hand-eye calibration proof (sim)",
        "kind": "calibration",
        "description": (
            "A camera placed at random near the laptop, a lens handed over slightly wrong, the elbow 17 degrees "
            "off and the other joints up to 3, and the paddle mounted 2 mm and 2 degrees wrong. The arm shows "
            "its STag markers to the camera in ~50 poses; one fit solves the camera pose, the joint offsets and "
            "the mount from every marker corner, and is scored against the truth it never saw."),
        "args": {"seed": 0},
        "command": lambda a: [PY, "-m", "hw.handeye", "--sim", "--seed", str(a["seed"])],
    },
    "lens": {
        "title": "Lens calibration proof (sim, ChArUco)",
        "kind": "calibration",
        "description": (
            "The printed ChArUco board at 80 random poses across the frame, 0.3-0.8 m away, through a known "
            "14-coefficient lens. The fit is scored on reprojection, the recovered lens, and the 3D position of "
            "held-out boards against the truth."),
        "args": {"seed": 0},
        "command": lambda a: [PY, "-m", "hw.sim_intrinsics", "--seed", str(a["seed"])],
    },
    "render_policy": {
        "title": "Watch a policy (sim)",
        "kind": "policy",
        "description": (
            "Plays a keep-up policy version in the randomised world from a given drop seed, streaming the sim "
            "live and saving the video for the Policies page."),
        "args": {"version": 5, "seed": 10000, "episodes": 2},
        "command": lambda a: [PY, "-m", "lab.render_policy", "--run", f"v{a['version']}", "--seed", str(a["seed"]),
                              "--episodes", str(a["episodes"])],
    },
    "policy_curves": {
        "title": "Rebuild training curves from checkpoints",
        "kind": "training",
        "description": (
            "v1-v7 were trained without logging, so their loss curves are gone. This evaluates every 2M-step "
            "checkpoint (8 randomised drops, up to 120 s each) and records how long the policy kept the ball up: "
            "a performance-over-training curve for each version."),
        "args": {"every": 2000000, "episodes": 8},
        "command": lambda a: [PY, "-m", "lab.policy_curves", "--every", str(a["every"]), "--episodes", str(a["episodes"])],
    },
}
