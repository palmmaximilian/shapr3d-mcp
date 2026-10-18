"""frame_fe: beam FE of tube frames against closed-form results."""

from pathlib import Path

import numpy as np
import pytest

from shapr3d_mcp import frame_fe as fe

E = fe.STEEL["E"]
SQ = fe.section([100, 5])
I = SQ["Iy"]


def _cantilever(L=1000.0, **kw):
    return {"members": [{"name": "Tube", "p0": [0, 0, 0], "p1": [L, 0, 0], "section": [100, 5]}],
            "supports": [{"point": [0, 0, 0], "fix": "all"}], **kw}


def test_section_props_en10219():
    assert SQ["A"] == pytest.approx(100 ** 2 - 90 ** 2)
    assert SQ["Iy"] == pytest.approx((100 ** 4 - 90 ** 4) / 12)
    r = fe.section([120, 60, 4])               # b 120 (local y), h 60 (local z)
    assert r["Iz"] > r["Iy"] and r["Wy"] == pytest.approx(r["Iy"] / 30)
    with pytest.raises(ValueError):
        fe.section([50, 30])


def test_cantilever_tip_deflection_and_stress():
    L, P = 1000.0, 2000.0
    R = fe.analyse(_cantilever(L, loads=[{"point": [L, 0, 0], "F": [0, 0, -P], "name": "tip"}]),
                   n_modes=0)
    tip = R.M.points["tip"]
    w = R.cases["load"].U[6 * tip + 2, 0]
    assert w == pytest.approx(-P * L ** 3 / (3 * E * I), rel=1e-6)
    s = fe.summary(R)
    assert s["max_stress"]["MPa"] == pytest.approx(P * L / SQ["Wy"], rel=0.02)
    assert s["max_stress"]["member"] == "Tube"
    assert s["stiffness"]["tip"]["z"] == pytest.approx(3 * E * I / L ** 3, rel=1e-3)


def test_cantilever_first_frequency():
    L = 1000.0
    R = fe.analyse(_cantilever(L, max_element_mm=50), n_modes=2)
    m_per_mm = SQ["A"] * fe.STEEL["rho"]                   # kg/mm
    f1 = 1.875104 ** 2 / (2 * np.pi) * np.sqrt(E * I / (m_per_mm * L ** 4) * 1000)
    # square tube: the two bending modes coincide
    assert R.modes[0][0] == pytest.approx(f1, rel=0.01)
    assert R.modes[1][0] == pytest.approx(f1, rel=0.01)


def test_tip_mass_frequency():
    L, kg = 1000.0, 50.0
    R = fe.analyse(_cantilever(L, masses=[{"point": [L, 0, 0], "kg": kg}]), n_modes=1)
    m_eff = kg + 0.2357 * SQ["A"] * L * fe.STEEL["rho"]     # Rayleigh tip-mass model
    f = np.sqrt(3 * E * I / L ** 3 * 1000 / m_eff) / (2 * np.pi)
    assert R.modes[0][0] == pytest.approx(f, rel=0.01)


def test_offset_load_point_gets_rigid_link():
    L, P = 1000.0, 1000.0
    R = fe.analyse(_cantilever(L, loads=[{"point": [L, 0, 200], "F": [0, 0, -P], "name": "top"}]),
                   n_modes=0)
    assert any(e[3] == "link" for e in R.M.el)
    w = R.cases["load"].U[6 * R.M.points["top"] + 2, 0]
    assert w == pytest.approx(-P * L ** 3 / (3 * E * I), rel=1e-3)


def test_rigid_body_check():
    R = fe.analyse(_cantilever(), n_modes=0)
    assert R.rigid_residual < 1e-9
    K = R.K.copy()
    K[0, 0] += 1e3 * np.abs(K).max()              # spring to ground: not stress-free
    assert fe.rigid_body_check(R.M, K) > 1e-3


def _portal(H=1500.0, Lb=1000.0, beam=(100, 5), col=(100, 5)):
    return {"members": [
        {"name": "Column left", "p0": [0, 0, 0], "p1": [0, 0, H], "section": list(col)},
        {"name": "Column right", "p0": [Lb, 0, 0], "p1": [Lb, 0, H], "section": list(col)},
        {"name": "Beam", "p0": [0, 0, H], "p1": [Lb, 0, H], "section": list(beam)}],
        "supports": [{"point": [0, 0, 0], "fix": "all"}, {"point": [Lb, 0, 0], "fix": "all"}]}


def test_portal_frame_sway():
    H, Lb, P = 1500.0, 1000.0, 1000.0
    spec = _portal(H, Lb, beam=(120, 60, 4))       # b 120 horizontal, h 60 vertical
    spec["loads"] = [{"point": [0, 0, H], "F": [P, 0, 0], "name": "corner"}]
    R = fe.analyse(spec, n_modes=0)
    Ic, Ib = SQ["Iy"], fe.section([120, 60, 4])["Iy"]
    k = (Ib / Lb) / (Ic / H)
    # slope-deflection, fixed bases, rigid joints, no axial strain
    delta = P * H ** 3 / (12 * E * Ic) * (2 + 3 * k) / (1 + 6 * k)
    u = R.cases["load"].U[6 * R.M.points["corner"], 0]
    assert u == pytest.approx(delta, rel=0.01)
    assert not any(e[3] == "link" for e in R.M.el)     # corners coincide: no offsets


def test_portal_needs_supports():
    spec = _portal()
    spec["supports"] = [[0, 0, 0]]                 # one pin: mechanism
    spec["loads"] = [{"point": [0, 0, 1500], "F": [1, 0, 0]}]
    with pytest.raises(ValueError, match="mechanism|singular"):
        fe.analyse(spec, n_modes=0)


def test_sweep_envelope_and_joint_offsets():
    # stand: post on a base rail, arm on top; post end sits 50 mm off the rail axis
    spec = {"members": [
        {"name": "Rail", "p0": [-400, 0, 0], "p1": [400, 0, 0], "section": [100, 5]},
        {"name": "Post", "p0": [0, 0, 50], "p1": [0, 0, 800], "section": [100, 5]},
        {"name": "Plate", "p0": [0, 0, 800], "p1": [0, 0, 820], "section": "rigid"}],
        "supports": [[-400, -300, 0], [-400, 300, 0], [400, -300, 0], [400, 300, 0]],
        "sweeps": [{"point": [0, 0, 820], "Fxy": 500, "Fz": 800, "Mxy": 300, "Mz": 200,
                    "case": "stop", "name": "base"}]}
    with pytest.raises(ValueError):                 # supports far off any member: offsets ok,
        fe.analyse({**spec, "supports": []})       # but no supports at all is an error
    R = fe.analyse(spec, n_modes=0)
    assert any(e[3] == "link" for e in R.M.el)       # post foot joined to the rail
    c = R.cases["stop"]
    assert c.U.shape[1] == 12 * 12 * 2 * 2
    s = fe.summary(R)
    assert s["max_stress"]["case"] == "stop"
    assert set(s["stiffness"]["base"]) == set(fe.DOF)
    # envelope >= any single direction of the same magnitudes
    single = fe.analyse({**spec, "sweeps": [], "loads": [
        {"point": [0, 0, 820], "F": [500, 0, 800], "M": [300, 0, 200], "name": "base"}]}, n_modes=0)
    assert s["max_stress"]["MPa"] >= fe.summary(single)["max_stress"]["MPa"] - 1e-9


def test_plots_written(tmp_path):
    spec = _portal()
    spec["loads"] = [{"point": [0, 0, 1500], "F": [1000, 0, 0], "name": "corner"}]
    spec["masses"] = [{"point": [500, 0, 1500], "kg": 20}]
    R = fe.analyse(spec)
    paths = fe.plot_all(R, tmp_path, "portal")
    assert set(paths) == {"stress", "deformation", "modes"}
    for p in paths.values():
        assert (tmp_path / p).exists() and (tmp_path / p).stat().st_size > 10_000


def test_analyse_frame_tool():
    from shapr3d_mcp import server
    spec = _portal()
    res = server.analyse_frame(spec["members"], spec["supports"],
                               loads=[{"point": [0, 0, 1500], "F": [1000, 0, 0], "name": "corner"}],
                               masses=[{"point": [500, 0, 1500], "kg": 20}],
                               name="frame_test/portal")
    assert res["max_stress"]["member"].startswith("Column")
    assert res["stiffness"]["corner"]["x"] > 0 and len(res["eigenfrequencies_Hz"]) == 3
    assert all(Path(p).exists() for p in res["plots"].values())
