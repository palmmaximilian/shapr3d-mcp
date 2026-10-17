"""Drawing engine: numbers on the sheet must equal the solid's geometry."""
from pathlib import Path

import pytest
from build123d import Box, Cylinder, Pos, export_step

from shapr3d_mcp import drawing as D
from shapr3d_mcp import server


def plate_with_holes():
    b = Box(190, 220, 20)
    for x, y in ((-60.1, -60.1), (60.1, 60.1), (60.1, -60.1)):
        b -= Pos(x, y, 0) * Cylinder(3.4, 30)
    b -= Pos(60, 10, 0) * Cylinder(4, 30)
    return b


def test_hole_centre_is_on_the_axis_not_the_half_face():
    """Regression: Face.center() of a half-cylinder bore is off by the radius."""
    hs = D.holes(plate_with_holes())
    got = {(round(h["x"], 2), round(h["y"], 2), h["d"]) for h in hs}
    assert (60.0, 10.0, 8.0) in got
    assert (-60.1, -60.1, 6.8) in got and (60.1, 60.1, 6.8) in got
    assert len(hs) == 4


def test_pick_scale_uses_standard_reductions():
    assert D.pick_scale([100, 100], [200, 200]) == 1
    assert D.pick_scale([300, 100], [200, 200]) == 2
    assert D.pick_scale([1500, 900], [240, 180]) == 10


def test_part_sheet_registers_measured_overall_dimensions(tmp_path):
    info = D.TitleInfo(title="Testplatte", number="T-1")
    sh = D.part_sheet(plate_with_holes(), info)
    dims = {d["bezeichnung"]: d["wert"] for d in sh.dims}
    assert dims["Länge X"] == pytest.approx(190, abs=1e-6)
    assert dims["Breite Y"] == pytest.approx(220, abs=1e-6)
    assert dims["Höhe Z"] == pytest.approx(20, abs=1e-6)
    holes = [d["wert"] for d in sh.dims if d["bezeichnung"].startswith("Bohrung")]
    assert len(holes) == 4
    # coordinates from the min corner: hole at (60, 10) -> (155, 120)
    assert any(abs(h[0] - 155) < 1e-6 and abs(h[1] - 120) < 1e-6 for h in holes)
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(tmp_path / "t.pdf") as pdf:
        sh.save(pdf)
    assert (tmp_path / "t.pdf").stat().st_size > 1000


def test_dimension_text_matches_value():
    sh = D.Sheet(D.TitleInfo(title="x", number="x"))
    D.dim(sh, (0, 0), (50, 0), 5, D.mm(123.45), register=("L", 123.45))
    assert sh.dims[0]["text"] == "123,5" and sh.dims[0]["wert"] == 123.45


def test_create_drawing_tool_writes_pdf(tmp_path, monkeypatch):
    step = tmp_path / "platte.step"
    export_step(plate_with_holes(), str(step))
    res = server.create_drawing(str(step), title="Platte", drawing_no="P-1", material="S235JR")
    assert Path(res["pdf"]).exists()
    assert res["scale"].startswith("1:")
    assert any(d["bezeichnung"] == "Länge X" and abs(d["wert"] - 190) < 1e-6 for d in res["dimensions"])
