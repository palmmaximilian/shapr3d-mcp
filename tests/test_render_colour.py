"""Colour render_preview: one colour per assembly part."""

import colorsys
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from shapr3d_mcp import server

WORKDIR = Path(os.environ["SHAPR3D_MCP_WORKDIR"])

TWO_PART_SCRIPT = """
a = Box(20, 20, 20)
a.label = "base"
b = Pos(30, 0, 0) * Cylinder(8, 25)
b.label = "post"
result = Compound(children=[a, b])
"""


def _hues(png: Path) -> set[int]:
    """30-degree hue bins that hold at least 2 % of the coloured pixels."""
    im = Image.open(png).convert("RGB")
    im.thumbnail((400, 400))
    bins: dict[int, int] = {}
    total = 0
    for r, g, b in np.asarray(im).reshape(-1, 3):
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        if s > 0.25 and v > 0.2:  # skip white background, grey, outlines
            bins[int(h * 12) % 12] = bins.get(int(h * 12) % 12, 0) + 1
            total += 1
    return {k for k, n in bins.items() if n >= 0.02 * total}


def test_render_preview_colours_two_parts():
    server.create_model("two_parts", TWO_PART_SCRIPT)
    server.render_preview("two_parts.step", legend=False)
    hues = _hues(WORKDIR / "two_parts_preview.png")
    assert len(hues) >= 2, hues


def test_colour_groups_explicit_colours_and_palette():
    from build123d import Box, Compound, Pos

    a, b, c = Box(1, 1, 1), Pos(3, 0, 0) * Box(1, 1, 1), Pos(6, 0, 0) * Box(1, 1, 1)
    a.label, b.label, c.label = "Cart_frame", "UR16e arm", "castor"
    groups = server._colour_groups(Compound(children=[a, b, c]), "asm",
                                   colors={"ur16e": "ff0000"},
                                   palette=["#00ff00", [0, 0, 255]])
    assert [(n, rgb) for n, _, rgb in groups] == [
        ("Cart_frame", (0.0, 1.0, 0.0)),
        ("UR16e arm", (1.0, 0.0, 0.0)),
        ("castor", (0.0, 0.0, 1.0)),
    ]
    with pytest.raises(ValueError, match="palette"):
        server._colour_groups(Compound(children=[a, b]), "asm", palette="no_such_map")
