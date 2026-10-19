"""Write a 3MF holding one object made of several coloured parts.

    write_3mf("out/part.3mf", {"black": (verts, faces), "white": (verts, faces)})

A multi-colour print in Bambu Studio is one object whose parts each get a
filament. build123d's Mesher can write 3MF, but it rejects any mesh with a
stray edge - and the tagged paddle's inlays, traced from a marker image, come
out with a handful (15 in 80 000) that every slicer repairs on import. So this
writes the 3MF directly: each part as a mesh object, and one object made of
them all as components, which is what makes Bambu Studio load them as parts of
one object rather than as separate objects. Units are millimetres.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>"""

RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"""

COLOURS = {"black": "#1A1A1AFF", "white": "#F2F2F2FF"}


def _mesh_xml(verts: np.ndarray, faces: np.ndarray) -> str:
    v = "\n".join(f'<vertex x="{x:.5f}" y="{y:.5f}" z="{z:.5f}"/>' for x, y, z in verts)
    t = "\n".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in faces)
    return f"<mesh><vertices>\n{v}\n</vertices><triangles>\n{t}\n</triangles></mesh>"


def write_plain_3mf(path: Path | str, parts: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """The plainest 3MF there is: one mesh object per part, each its own
    build item, no materials, no components. Every slicer reads this; Bambu
    Studio asks whether to load the parts as one object - answer yes."""
    objs, items = [], []
    for k, (n, (v, f)) in enumerate(parts.items()):
        objs.append(f'<object id="{k + 1}" name="{n}" type="model">{_mesh_xml(v, f)}</object>')
        items.append(f'<item objectid="{k + 1}"/>')
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n'
        "<resources>\n" + "\n".join(objs) + "\n</resources>\n<build>" + "".join(items) + "</build>\n</model>\n"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("3D/3dmodel.model", model)


def write_3mf(path: Path | str, parts: dict[str, tuple[np.ndarray, np.ndarray]], name: str = "part") -> None:
    colours = "".join(f'<base name="{n}" displaycolor="{COLOURS.get(n, "#808080FF")}"/>' for n in parts)
    objs, comps = [], []
    for k, (n, (v, f)) in enumerate(parts.items()):
        oid = k + 2
        objs.append(f'<object id="{oid}" name="{n}" type="model" pid="1" pindex="{k}">{_mesh_xml(v, f)}</object>')
        comps.append(f'<component objectid="{oid}"/>')
    top = len(parts) + 2
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n'
        f'<resources><basematerials id="1">{colours}</basematerials>\n'
        + "\n".join(objs)
        + f'\n<object id="{top}" name="{name}" type="model"><components>{"".join(comps)}</components></object>'
        f'</resources>\n<build><item objectid="{top}"/></build>\n</model>\n'
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("3D/3dmodel.model", model)
