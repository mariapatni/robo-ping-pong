"""Add the table to a scene, at load time, via MjSpec.

The vendored MJCF in sim/models/so101 is upstream mujoco_menagerie and is left
alone so fixes can be pulled from it. Anything this project adds to the world is
added here, to the spec, before compile.
"""

from __future__ import annotations

import mujoco

from .config import TableConfig

TABLE_GEOM = "table"


def add_table(spec: mujoco.MjSpec, cfg: TableConfig) -> mujoco.MjSpec:
    """Drop a table slab into the world with its top at z = 0.

    The arm's base is at the origin, i.e. bolted to the middle of the near short
    edge, so the slab is centred half a length out in +x. The scene's ground
    plane is pushed down to the real floor height so the arm is not standing on
    two surfaces at once.
    """
    floor = spec.geom("floor")
    if floor is not None:
        floor.pos = [0.0, 0.0, -cfg.height]

    spec.worldbody.add_geom(
        name=TABLE_GEOM,
        pos=[cfg.half_length, 0.0, -0.5 * cfg.thickness],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[cfg.half_length, cfg.half_width, 0.5 * cfg.thickness],
        rgba=[0.10, 0.35, 0.28, 1.0],
        condim=3,
        friction=list(cfg.friction),
        solref=list(cfg.contact_solref),
        # The table, not the ball, owns the bounce: priority means this geom's
        # solref and friction are used for every contact it takes part in, so a
        # paddle added later can carry its own, different restitution.
        priority=1,
    )
    return spec
