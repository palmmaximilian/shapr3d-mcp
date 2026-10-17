"""mass_properties: per-part mass, CoG and inertia of labelled assemblies,
vendor catalogue masses and the UR payload check."""

import os
from pathlib import Path

import numpy as np
import pytest
from build123d import Box, Compound, Cylinder, Pos, Rot, export_step

from shapr3d_mcp import kernel, mass, server

WORKDIR = Path(os.environ["SHAPR3D_MCP_WORKDIR"])
REPO_ASM = Path(__file__).resolve().parents[1] / "models" / "ur10e_rg2_assembly.step"


def _asm():
    # 10x20x30 steel box with its centre at (100, 0, -50); two moved/rotated
    # copies of an aluminium cylinder (r 5, h 40) at x = +-200, z = -100.
    cyl = Cylinder(5, 40)
    return Compound(label="asm", children=[
        Compound(label="Block", children=[Pos(100, 0, -50) * Box(10, 20, 30)]),
        Compound(label="Pin 1", children=[Pos(-200, 0, -100) * cyl]),
        Compound(label="Pin 2", children=[Pos(200, 0, -100) * Rot(0, 0, 90) * cyl]),
        Compound(label="UR10e wrist", children=[Box(5, 5, 5)]),
    ])


DENS = {"Block": "steel", "Pin": "aluminium"}


def test_box_mass_cog_and_inertia_about_origin():
    out = mass.mass_properties(_asm(), {"Block": "steel"}, exclude=["Pin", "wrist"])
    p = out["parts"][0]
    m = 6000 * 7.85e-6  # mm3 * kg/mm3
    assert p["mass_kg"] == pytest.approx(m, rel=1e-4)
    assert p["cog_mm"] == [100.0, 0.0, -50.0]
    # about the CoG: m(b^2+c^2)/12; about origin add m(|r|^2 - r_i r_j)
    r = np.array([100, 0, -50]) * 1e-3
    ic = m / 12 * np.array([20**2 + 30**2, 10**2 + 30**2, 10**2 + 20**2]) * 1e-6
    io = np.diag(ic) + m * (r @ r * np.eye(3) - np.outer(r, r))
    got = p["inertia_origin_kgm2"]
    assert got["Ixx"] == pytest.approx(io[0, 0], rel=1e-3)
    assert got["Iyy"] == pytest.approx(io[1, 1], rel=1e-3)
    assert got["Izz"] == pytest.approx(io[2, 2], rel=1e-3)
    assert got["Ixz"] == pytest.approx(io[0, 2], rel=1e-3)
    assert got["Ixy"] == 0.0 and got["Iyz"] == 0.0
    tot = out["total"]
    assert tot["inertia_cog_kgm2"]["Ixx"] == pytest.approx(ic[0], rel=1e-3)
    assert tot["inertia_cog_kgm2"]["Ixz"] == 0.0


def test_totals_combine_parts_and_copies_integrate_once(monkeypatch):
    calls = []
    orig = kernel.Measures.gprops

    def spy(self, solid):
        entry = self._entry(solid)
        if "gprops" not in entry:
            calls.append(1)
        return orig(self, solid)

    monkeypatch.setattr(kernel.Measures, "gprops", spy)
    out = mass.mass_properties(_asm(), DENS, exclude=["wrist"])
    assert len(calls) == 2  # box + one cylinder for both placed pins
    names = [p["part"] for p in out["parts"]]
    assert names == ["Block", "Pin 1", "Pin 2"] and out["excluded"] == ["UR10e wrist"]
    m_pin = np.pi * 25 * 40 * 2.7e-6
    m_box = 6000 * 7.85e-6
    assert out["total"]["mass_kg"] == pytest.approx(m_box + 2 * m_pin, abs=1e-3)
    cog = out["total"]["cog_mm"]
    assert cog[0] == pytest.approx(100 * m_box / (m_box + 2 * m_pin), abs=0.1)
    assert cog[2] == pytest.approx((-50 * m_box - 200 * m_pin) / (m_box + 2 * m_pin), abs=0.1)
    # the rotated copy has the same mass and mirrored CoG
    p1, p2 = out["parts"][1:]
    assert p1["mass_kg"] == p2["mass_kg"] and p1["cog_mm"][0] == -p2["cog_mm"][0]


def test_vendor_mass_scales_density_and_longest_key_wins():
    dens = {"*": "steel", "Pin": "aluminium", "Pin 2": 1.0}
    out = mass.mass_properties(_asm(), dens, vendor_masses={"Block": 0.5},
                               exclude=["wrist"])
    block, pin1, pin2 = out["parts"]
    assert block["mass_kg"] == 0.5 and block["material"] == "vendor mass 0.5 kg"
    assert block["density_g_cm3"] == pytest.approx(0.5 / 6000 * 1e6, rel=1e-3)
    assert block["cog_mm"] == [100.0, 0.0, -50.0]
    assert pin1["density_g_cm3"] == 2.7 and pin2["density_g_cm3"] == 1.0


def test_unassigned_part_and_bad_material_raise():
    with pytest.raises(ValueError, match="No material for part.*Pin 1"):
        mass.mass_properties(_asm(), {"Block": "steel"}, exclude=["wrist"])
    with pytest.raises(ValueError, match="Unknown material 'unobtainium'"):
        mass.mass_properties(_asm(), {"*": "unobtainium"})
    assert mass.density("Aluminum") == 2.7 and mass.density("PA12") == 0.95
    assert mass.density(1.2) == 1.2 and mass.density("7.9") == 7.9


def test_payload_curve_and_robot_check():
    assert mass.payload_limit("UR10e", 50) == (12.5, True)
    assert mass.payload_limit("ur10e", 175)[0] == pytest.approx(11.6)
    assert mass.payload_limit("UR10e", 400) == (9.6, False)
    with pytest.raises(ValueError, match="No payload curve"):
        mass.payload_limit("UR99", 0)
    heavy = mass.robot_check("UR10e", 11.0, [0, 0, -250])
    assert not heavy["ok"] and "exceeds" in heavy["warnings"][0]
    near = mass.robot_check("UR10e", 11.5, [0, 0, -100])
    assert near["ok"] and "within 10 %" in near["warnings"][0]
    far = mass.robot_check("UR10e", 1.0, [0, 0, -500])
    assert not far["ok"] and "beyond" in far["warnings"][0]


def test_robot_excludes_its_own_parts_by_default():
    out = mass.mass_properties(_asm(), DENS, robot="UR10e")
    assert out["excluded"] == ["UR10e wrist"]
    rc = out["robot_check"]
    assert rc["ok"] and rc["allowed_at_offset_kg"] == 12.5 and rc["warnings"] == []


def test_tool_reads_labelled_step():
    path = WORKDIR / "mass_asm.step"
    export_step(_asm(), str(path))
    out = server.mass_properties("mass_asm.step", DENS, robot="UR10e")
    assert out["file"] == str(path)
    # STEP labels come back with underscores; substring matching ignores that
    assert [p["part"] for p in out["parts"]] == ["Block", "Pin_1", "Pin_2"]
    assert out["total"]["mass_kg"] == pytest.approx(0.0471 + 2 * 0.0085, abs=1e-3)


@pytest.mark.skipif(not REPO_ASM.exists(), reason="build models/ur10e_rg2_assembly.step first")
def test_ur10e_rg2_beam_payload():
    shape = server._import_cached(REPO_ASM)
    out = mass.mass_properties(
        shape,
        {"Bracket": "PA12", "Carriage": "PA12", "Beam": "aluminium",
         "Screws": "steel", "T-nuts": "steel", "Dowels": "steel"},
        {"RG2": 0.78, "QC robot side": 0.06}, robot="UR10e")
    tot = out["total"]
    assert 2.5 < tot["mass_kg"] < 5.0
    assert abs(tot["cog_mm"][0]) < 5 and abs(tot["cog_mm"][1]) < 5  # symmetric beam
    assert -150 < tot["cog_mm"][2] < -50
    assert out["robot_check"]["ok"]
