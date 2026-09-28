"""Add the table, its surface, its corner markers and its net, via MjSpec.

The vendored MJCF in sim/models/so101 is upstream mujoco_menagerie and is left
alone so fixes can be pulled from it. Anything this project adds to the world is
added here, to the spec, before compile.

The visual dressing in this module is not decoration. The real table is waxed
wood and the real corners will carry fiducials, and both of those are what the
cameras will actually see; a scene that is flat green with unmarked edges would
be training a perception stack against a background that does not exist.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .config import MarkerConfig, NetConfig, SurfaceConfig, TableConfig

TABLE_GEOM = "table"
WOOD_TEXTURE = "table_wood"
WOOD_MATERIAL = "table_wood"
MARKER_GEOMS = ("marker_near_left", "marker_near_right", "marker_far_right", "marker_far_left")
NET_GEOM = "net"
NET_POST_GEOMS = ("net_post_left", "net_post_right")


# -- the surface --------------------------------------------------------------


def wood_texture(cfg: SurfaceConfig) -> np.ndarray:
    """A procedural waxed-wood grain, returned as (n, n, 3) uint8.

    Rows run along +x (the direction of play), columns across the table, so the
    grain lines run lengthwise as the planks of a butcher-block island do. Three
    layers, which is the minimum that stops reading as wallpaper: grain lines
    whose spacing wanders along the plank, thin streaks that vary only across
    the grain, and dark seams between planks.

    Deterministic in `cfg.seed`, so the same config renders the same table on
    every machine and a rendered-image test is meaningful.
    """
    n = int(cfg.texture_px)
    rng = np.random.default_rng(cfg.seed)
    u = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]  # along the grain
    v = np.linspace(0.0, 1.0, n, endpoint=False)[None, :]  # across it

    # The grain lines are not straight: they wander slowly along the plank.
    wander = 0.05 * np.sin(2 * np.pi * (1.3 * u + 0.21)) + 0.025 * np.sin(2 * np.pi * (2.7 * u + 0.73))
    g = v + wander
    lines = 0.5 + 0.5 * np.sin(2 * np.pi * cfg.grain_lines * g + 0.9 * np.sin(2 * np.pi * 3.0 * g))
    lines = lines ** cfg.grain_contrast

    # Fine streaks: noise that varies across the grain only, so it smears into
    # lengthwise fibres instead of looking like sensor noise.
    coarse = rng.random(max(4, n // 4))
    streaks = np.repeat(coarse, int(np.ceil(n / coarse.size)))[:n][None, :]
    speckle = rng.random((n, n))

    t = np.clip(0.62 * lines + 0.26 * streaks + 0.12 * speckle, 0.0, 1.0)

    # Plank seams, a couple of pixels of near-black between boards.
    if cfg.planks >= 2:
        seam_w = max(1, n // 256)
        for k in range(1, int(cfg.planks)):
            c = int(round(k * n / cfg.planks))
            t[:, max(0, c - seam_w) : c + seam_w] = 1.0

    light = np.asarray(cfg.light_rgb, dtype=float)
    dark = np.asarray(cfg.dark_rgb, dtype=float)
    rgb = light[None, None, :] + t[..., None] * (dark - light)[None, None, :]
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def add_wood_material(spec: mujoco.MjSpec, cfg: SurfaceConfig) -> str:
    """Register the generated grain as a texture + material. Returns its name.

    The bitmap is handed to the spec as raw bytes rather than written to a PNG:
    there is then no generated asset on disk to drift out of step with the
    config, and nothing for a test to have to clean up.
    """
    tex = spec.add_texture()
    tex.name = WOOD_TEXTURE
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    img = wood_texture(cfg)
    tex.height, tex.width, tex.nchannel = img.shape[0], img.shape[1], 3
    tex.data = img.tobytes()

    mat = spec.add_material()
    mat.name = WOOD_MATERIAL
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = WOOD_TEXTURE
    mat.texrepeat = list(cfg.repeat)
    mat.texuniform = False
    mat.specular = cfg.specular
    mat.shininess = cfg.shininess
    mat.reflectance = cfg.reflectance
    return WOOD_MATERIAL


def add_table(
    spec: mujoco.MjSpec,
    cfg: TableConfig,
    surface: SurfaceConfig | None = None,
) -> mujoco.MjSpec:
    """Drop a table slab into the world with its top at z = 0.

    The arm's base is at the origin, i.e. bolted to the middle of the near short
    edge, so the slab is centred half a length out in +x. The scene's ground
    plane is pushed down to the real floor height so the arm is not standing on
    two surfaces at once.

    `surface` only changes how it looks, never how it behaves: the bounce comes
    from `cfg`, so a test may turn the wood off without perturbing any physics.
    """
    surface = surface or SurfaceConfig()
    floor = spec.geom("floor")
    if floor is not None:
        floor.pos = [0.0, 0.0, -cfg.height]

    geom = spec.worldbody.add_geom(
        name=TABLE_GEOM,
        pos=[cfg.half_length, 0.0, -0.5 * cfg.thickness],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[cfg.half_length, cfg.half_width, 0.5 * cfg.thickness],
        rgba=[1.0, 1.0, 1.0, 1.0] if surface.wood else list(surface.plain_rgba),
        condim=3,
        friction=list(cfg.friction),
        solref=list(cfg.contact_solref),
        # The table, not the ball, owns the bounce: priority means this geom's
        # solref and friction are used for every contact it takes part in, so a
        # paddle added later can carry its own, different restitution.
        priority=1,
    )
    if surface.wood:
        geom.material = add_wood_material(spec, surface)
    return spec


# -- the bounds of play -------------------------------------------------------


def corner_positions(cfg: TableConfig, markers: MarkerConfig) -> list[tuple[float, float, float]]:
    """Centres of the four corner markers, going round the table.

    Order matches MARKER_GEOMS and MarkerConfig.rgba: near +y, near -y, far -y,
    far +y. Inset from both edges so the whole disc is on the surface, which is
    also how a printed fiducial has to be taped down for its outer border to be
    detectable.
    """
    d = markers.inset
    x0, x1 = d, cfg.length - d
    y = cfg.half_width - d
    z = markers.thickness
    return [(x0, y, z), (x0, -y, z), (x1, -y, z), (x1, y, z)]


def add_corner_markers(
    spec: mujoco.MjSpec, cfg: TableConfig, markers: MarkerConfig | None = None
) -> mujoco.MjSpec:
    """Four coloured discs at the corners: the bounds of play, and the stand-in
    for the calibration fiducials on the real island. Non-colliding."""
    markers = markers or MarkerConfig()
    if not markers.enabled:
        return spec
    for name, pos, rgba in zip(MARKER_GEOMS, corner_positions(cfg, markers), markers.rgba):
        spec.worldbody.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=list(pos),
            size=[markers.radius, markers.thickness, 0.0],
            rgba=list(rgba),
            contype=0,
            conaffinity=0,
        )
    return spec


def add_net(spec: mujoco.MjSpec, cfg: TableConfig, net: NetConfig | None = None) -> mujoco.MjSpec:
    """The centre divider at x = length/2, plus its two posts.

    See NetConfig for why the height is 60 mm and why collision is off by
    default. When collision is on the net carries its own, very lossy contact
    and takes priority, so a clipped ball drops rather than rebounding like it
    hit a wall.
    """
    net = net or NetConfig()
    if not net.enabled:
        return spec

    half_span = cfg.half_width + net.overhang
    kw: dict = {} if net.collide else {"contype": 0, "conaffinity": 0}
    if net.collide:
        kw = {"condim": 3, "solref": list(net.contact_solref), "priority": 2}
    spec.worldbody.add_geom(
        name=NET_GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[cfg.half_length, 0.0, 0.5 * net.height],
        size=[0.5 * net.thickness, half_span, 0.5 * net.height],
        rgba=list(net.rgba),
        **kw,
    )
    for name, sign in zip(NET_POST_GEOMS, (1.0, -1.0)):
        spec.worldbody.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[cfg.half_length, sign * half_span, 0.5 * net.height],
            size=[net.post_radius, 0.5 * net.height, 0.0],
            rgba=list(net.post_rgba),
            contype=0,
            conaffinity=0,
        )
    return spec
