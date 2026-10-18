"""UR robot tools: DH kinematics (FK/IK round trip), posing the vendor STEP,
capsule reach study, real-CAD collision check and cart tipping, checked
against hand calculations and the printer cart study they came from
(models/printer_robot, frozen in tests/data/printer_cart.json)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from build123d import Box, Compound, Pos, export_step

from shapr3d_mcp.robots import reach as R
from shapr3d_mcp.robots import tipping as TP
from shapr3d_mcp.robots import tools as T
from shapr3d_mcp.robots import ur_pose as UP
from shapr3d_mcp.robots.ur_kinematics import (MODELS, fk, fk_all, get_model, ik, matrix_rpy, pose,
                                              rpy_matrix, wrap)

REPO = Path(__file__).resolve().parents[1]
CART = json.loads((Path(__file__).parent / "data" / "printer_cart.json").read_text())
PRINTER_ROBOT = REPO / "models" / "printer_robot"
UR16E_STEP = REPO / "models" / "vendor" / "UR16e.step"
G = 9.81

# the cleaning cart's hose attachment: nozzle 410 mm along flange z, 73.15 mm off-axis
HOSE_TOOL = {"tcp": {"xyz": [-73.15, 0, 410]}, "capsules": [
    {"p0": [-70, 0, 5], "p1": [40, 0, 5], "r": 5}, {"p0": [0, -40, 5], "p1": [0, 40, 5], "r": 5},
    {"p0": [-73.15, 0, -200], "p1": [-73.15, 0, -40], "r": 35}]}
HOSE_MASSES = [{"kg": 1.0, "xyz": [-73.15 / 2, 0, 205]}, {"kg": 3.0, "xyz": [-73.15, 0, 0]}]


# ---- kinematics -------------------------------------------------------------

@pytest.mark.parametrize("name", ["UR3e", "UR5e", "UR10e", "UR16e", "UR20"])
def test_ik_returns_all_8_solutions_that_round_trip(name):
    rng = np.random.default_rng(1)
    checked = 0
    while checked < 20:
        q = wrap(rng.uniform(-np.pi, np.pi, 6))
        if abs(np.sin(q[4])) < 0.2:
            continue  # near the wrist singularity the 8 branches degenerate
        T06 = fk(name, q)
        sols = ik(name, T06)
        if len(sols) < 8:
            continue  # elbow branch out of reach for one shoulder side
        assert len(sols) == 8
        for s in sols:
            assert np.allclose(fk(name, s), T06, atol=1e-6)
        assert min(np.abs(wrap(s - q)).max() for s in sols) < 1e-6  # original among them
        # 8 distinct configurations
        assert len({tuple(np.round(s, 4)) for s in sols}) == 8
        checked += 1


def test_known_ur16e_poses_by_hand():
    m = get_model("ur16e")
    # zero position: arm stretched horizontally along -x
    z = fk(m, np.zeros(6))
    assert np.allclose(z[:3, 3], [m.a2 + m.a3, -(m.d4 + m.d6), m.d1 - m.d5])
    assert np.allclose(z[:3, 3], [-838.4, -290.7, 60.85])
    # straight up (the vendor STEP home pose)
    up = fk(m, np.radians([0, -90, 0, -90, 0, 0]))
    assert np.allclose(up[:3, 3], [0, -290.7, 180.7 + 478.4 + 360 + 119.85])
    assert np.allclose(up[:3, 2], [0, -1, 0])
    assert np.allclose(UP.home_q("UR16e"), np.radians([0, -90, 0, -90, 0, 0]))
    # a general pose comes back among its own IK branches
    q = np.radians([30, -100, 80, -70, -90, 15])
    assert any(np.allclose(s, q, atol=1e-9) for s in ik(m, fk(m, q)))


def test_models_and_pose_helpers():
    assert set(MODELS) >= {"UR3e", "UR5e", "UR10e", "UR16e", "UR20"}
    assert get_model("UR 10e").name == "UR10e"
    with pytest.raises(ValueError):
        get_model("UR99")
    rpy = [10, -20, 135]
    assert np.allclose(matrix_rpy(rpy_matrix(rpy)), rpy)
    T = pose({"xyz": [1, 2, 3], "rpy_deg": [180, 0, 0]})
    assert np.allclose(T, np.diag([1, -1, -1, 1]) + np.c_[np.zeros((4, 3)), [1, 2, 3, 0]])
    rv = pose({"rotvec": [0, 0, np.pi / 2]})
    assert np.allclose(rv[:3, :3], rpy_matrix([0, 0, 90]))


def test_ur_kinematics_tool_fk_ik():
    out = T.ur_kinematics("UR16e", q_deg=[0, -90, 0, -90, 0, 0], tcp={"xyz": [0, 0, 100]},
                          base={"xyz": [1000, 0, 500]})
    assert out["flange"]["xyz"] == [1000.0, -290.7, 500 + 1138.95]
    assert out["tcp"]["xyz"] == [1000.0, -390.7, 500 + 1138.95]
    back = T.ur_kinematics("UR16e", tcp_pose={"xyz": out["tcp"]["xyz"], "rpy_deg": out["tcp"]["rpy_deg"]},
                           tcp={"xyz": [0, 0, 100]}, base={"xyz": [1000, 0, 500]})
    # straight up is wrist-singular (q5 = 0): no IK branch there
    assert back["solutions_q_deg"] == []
    q = [20, -80, 70, -60, -90, 10]
    f = T.ur_kinematics("UR10e", q_deg=q)
    sols = T.ur_kinematics("UR10e", tcp_pose=f["tcp"])["solutions_q_deg"]
    assert len(sols) == 8 and any(np.allclose(s, q, atol=1e-2) for s in sols)


# ---- reach (capsules vs boxes) ---------------------------------------------

def test_tcp_pose_offset_tool_matches_flange_geometry():
    tool = R.Tool.from_spec(HOSE_TOOL)
    tip = np.array([700.0, 1100, 1200])
    Ttcp = R.tcp_pose(tip, (0, 0, -1), np.radians(30))
    Tf = Ttcp @ np.linalg.inv(tool.tcp)
    x = np.array([np.cos(np.radians(30)), np.sin(np.radians(30)), 0])
    assert np.allclose(Tf[:3, 2], [0, 0, -1]) and np.allclose(Tf[:3, 0], x)
    assert np.allclose(Tf[:3, 3], tip + 73.15 * x + [0, 0, 410])


def test_box_obstacle_blocks_and_clears():
    tgt = {"name": "t", "xyz": [600, 0, 300]}
    free = R.reach_target("UR10e", np.eye(4), tgt, None, spin_step_deg=45)
    assert free is not None and np.isinf(free[0])
    # a slab right under the TCP: still reachable (down-pointing flange stays above it)
    lo, hi, _ = R.boxes([{"min": [400, -300, 0], "max": [800, 300, 280]}])
    above = R.reach_target("UR10e", np.eye(4), tgt, None, lo, hi, 45)
    assert above is not None and above[0] > 4
    # a wall enclosing the target: unreachable
    lo, hi, _ = R.boxes([{"min": [500, -100, 200], "max": [700, 100, 400]}])
    assert R.reach_target("UR10e", np.eye(4), tgt, None, lo, hi, 45) is None
    # out of reach entirely
    assert R.reach_target("UR3e", np.eye(4), {"xyz": [2000, 0, 0]}) is None


def test_printer_cart_all_117_targets_reached():
    """The cleaning cart result: every chamber/slot target reachable and
    clear of the printer boxes from the final base."""
    out = T.ur_reach_check("UR16e", CART["base"], CART["targets"], HOSE_TOOL, CART["boxes"],
                           spin_step_deg=15)
    assert out["targets"] == 117
    assert out["all_reached"], out["unreached"]
    assert out["min_clearance_mm"] > 4
    assert len(out["poses_q_deg"]) == 117


def test_base_search_finds_the_good_side():
    tg = [{"xyz": [700, 0, 200]}, {"xyz": [700, 200, 200]}]
    lo, hi = [-2000, -2000, -10], [2000, 2000, 0]   # floor
    out = T.ur_base_search("UR5e", tg, {"x": [0, 2000], "y": [0], "z": [300]},
                           obstacles=[{"min": lo, "max": hi}], spin_step_deg=45)
    assert out["bases_checked"] == 2 and out["bases_ok"] == 1
    assert out["best"][0]["base"]["xyz"] == [0.0, 0.0, 300.0]


def test_base_grid_axes():
    g = R.base_grid({"x": {"from": 0, "to": 100, "step": 50}, "y": [1, 2], "z": 5,
                     "yaw_deg": -90, "lean_deg": [0, 30], "lean_toward": [0, 0]})
    assert len(g) == 3 * 2 * 2
    assert g[-1] == {"xyz": [100.0, 2.0, 5.0], "yaw_deg": -90.0, "lean_deg": 30.0, "lean_toward": [0, 0]}


@pytest.mark.skipif(not (PRINTER_ROBOT / "reach.py").exists(), reason="printer_robot study not present")
def test_equivalent_to_printer_robot_reach_script():
    spec = importlib.util.spec_from_file_location("printer_reach", PRINTER_ROBOT / "reach.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    rng = np.random.default_rng(0)
    for _ in range(20):
        T06 = old.fk_all(rng.uniform(-3, 3, 6))[-1]
        a, b = old.ik(T06), ik("UR16e", T06)
        assert len(a) == len(b) and all(np.allclose(x, y) for x, y in zip(a, b))
    Tb_old = old.base_T(1150, 620, 1600, np.radians(30))
    Tb = R.base_T((1150, 620, 1600), -90, 30, lean_toward=old.CHAMBER_C)
    assert np.allclose(Tb_old, Tb)
    tool = R.Tool.from_spec(HOSE_TOOL)
    lo, hi, _ = R.boxes(old.BOXES)
    azs = np.radians(np.arange(0, 360, 30))
    for _, t in old.all_targets()[::10]:
        a = old.reach_target(Tb_old, t, azs)
        b = R.reach_target("UR16e", Tb, {"xyz": t}, tool, lo, hi, 30)
        assert (a is None) == (b is None)
        if a:
            assert np.isclose(a[0], b[0]) and np.allclose(a[1], b[1]) and a[2] == b[2]


# ---- posing the vendor STEP --------------------------------------------------

def _fake_vendor_step(model="UR16e"):
    """Seven 20 mm cubes at the link reference points of the home pose,
    labelled like the vendor file (order shuffled)."""
    refs = UP.link_reference_points(model)
    kids = [Compound(label=f"L{i}", children=[Pos(*refs[i]) * Box(20, 20, 20)]) for i in range(7)]
    return Compound(label="fake UR", children=[kids[i] for i in (3, 0, 6, 1, 5, 2, 4)])


def test_link_detection_and_posing_follow_fk():
    robot = _fake_vendor_step()
    assert UP.detect_link_map(robot, "UR16e") == [3, 0, 6, 1, 5, 2, 4]
    q = np.radians([30, -70, 50, -20, 60, 10])
    Tb = R.base_T((100, 200, 300), -90)
    parts = UP.posed_robot(robot, "UR16e", q, Tb)
    assert [p.label for p in parts] == [f"UR16e L{i}" for i in range(7)]
    F0, F = fk_all("UR16e", UP.home_q("UR16e")), fk_all("UR16e", q)
    for i, p in enumerate(parts):
        ref = UP.R_S2B @ np.r_[UP.link_reference_points("UR16e")[i], 1]   # home, base frame
        want = Tb @ F[i] @ np.linalg.inv(F0[i]) @ ref
        c = p.bounding_box().center()
        assert np.allclose([c.X, c.Y, c.Z], want[:3], atol=1e-6)
    # home pose with identity base = the STEP rotated into the DH base frame
    home = UP.posed_robot(robot, "UR16e", UP.home_q("UR16e"))
    c = home[6].bounding_box().center()
    assert np.allclose([c.X, c.Y, c.Z], (UP.R_S2B[:3, :3] @ UP.link_reference_points("UR16e")[6]), atol=1e-6)


def test_link_detection_rejects_wrong_robot():
    with pytest.raises(ValueError, match="link_map"):
        UP.detect_link_map(Compound(children=[Compound(label="a", children=[Box(1, 1, 1)])]), "UR16e")


def test_pose_step_and_collision_check_tools():
    from shapr3d_mcp import server

    robot_path = server.WORKDIR / "fake_ur16e.step"
    export_step(_fake_vendor_step(), str(robot_path))
    q = [0, -90, 0, -90, 0, 0]
    out = T.ur_pose_step("UR16e", str(robot_path), q, base={"xyz": [0, 0, 0], "yaw_deg": 0},
                         output_name="robots/posed")
    assert out["files"][0].endswith("posed.step") and len(out["parts"]) == 7
    flange = np.array(out["flange"]["xyz"])
    hit = Pos(*flange) * Box(200, 200, 200)
    away = Pos(2000, 0, 0) * Box(100, 100, 100)
    export_step(hit, str(server.WORKDIR / "hit.step"))
    export_step(away, str(server.WORKDIR / "away.step"))
    res = T.ur_collision_check("UR16e", str(robot_path), [q, [90, -90, 0, -90, 0, 0]],
                               ["hit.step", "away.step"], base={"xyz": [0, 0, 0], "yaw_deg": 0})
    r0, r1 = res["results"]
    assert r0["clash"] and r0["environment"]["hit.step"]["parts"]
    assert r0["environment"]["away.step"]["interference_mm3"] == 0
    # joint 1 turned 90 deg: the wrist swings off the box (cube centre ~150 mm off axis)
    assert r1["environment"]["hit.step"]["interference_mm3"] < r0["environment"]["hit.step"]["interference_mm3"]


@pytest.mark.skipif(not UR16E_STEP.exists(), reason="vendor UR16e.step not present")
def test_real_ur16e_step_link_detection():
    """The vendor file: detection reproduces the hand-made map of the
    printer cart study (ur16e_pose.LINK_OF)."""
    from shapr3d_mcp import server

    robot = server._import_cached(UR16E_STEP)
    got = dict(zip([c.label for c in robot.children], UP.detect_link_map(robot, "UR16e")))
    assert got == {"C-1000487": 0, "C-1000488": 1, "C-2007892": 2, "C-2007891": 3,
                   "C-1000491": 4, "C-1000492": 5, "C-2007000": 6}


# ---- tipping -------------------------------------------------------------------

# hand-calc cart: 100 kg point mass at (0, 0, 500), castors at x +-500, y +-300
BOX_CART = [{"kg": 100, "xyz": [0, 0, 500]}]
CASTORS = [[-500, -300], [500, -300], [500, 300], [-500, 300]]


def test_hand_force_by_hand():
    out = TP.hand_force(BOX_CART, CASTORS, 100, 1000)
    w = out["worst"]   # the short (y) lever: 0.3 m
    assert w["restoring_Nm"] == pytest.approx(100 * G * 0.3, abs=0.1)
    assert w["overturning_Nm"] == pytest.approx(100.0)
    assert w["safety_factor"] == pytest.approx(round(100 * G * 0.3 / 100, 2))


def test_acceleration_by_hand():
    allm = BOX_CART + [{"kg": 10, "xyz": [0, 0, 1000]}]
    out = TP.acceleration(allm, allm[1:], CASTORS, {"a2": 2.0})["a2"]
    assert out["safety_factor"] == pytest.approx(round(110 * G * 0.3 / (2.0 * 10 * 1.0), 2))
    # the edge towards +Y blocked (cart against a machine): x edges are the worst left
    out = TP.acceleration(allm, allm[1:], CASTORS, {"a2": 2.0}, ignore_edge_normals=[[0, 1], [0, -1]])["a2"]
    assert out["safety_factor"] == pytest.approx(round(110 * G * 0.5 / 20, 2))


def test_transport_by_hand():
    out = TP.transport(BOX_CART, CASTORS, trail_mm=34, handle_height_mm=1000, wheel_dia_mm=100,
                       thresholds_mm=[10], directions={"+X": (1, 0)})
    r = out["directions"][0]
    d, h, M = 0.466, 0.5, 100.0          # castor contacts trail 34 mm behind the axes
    assert r["cog_to_edge_mm"] == pytest.approx(466)
    assert r["stop_decel_m_s2"] == pytest.approx(G * d / h, abs=0.01)
    assert r["push_force_N"] == pytest.approx(M * G * d / 1.0, abs=0.1)
    assert r["slope_deg"] == pytest.approx(np.degrees(np.arctan(d / h)), abs=0.05)
    # blocked wheel: L = M v h about the edge, (L^2 / 2 I) = M g (sqrt(h^2+d^2) - h), I = M (h^2 + d^2)
    I_pivot = M * (h ** 2 + d ** 2)
    v = np.sqrt(2 * M * G * (np.hypot(h, d) - h) * I_pivot) / (M * h)
    assert r["impact_speed_m_s"] == pytest.approx(v, abs=0.006)
    # 10 mm step, R 50: tan = 0.75 at the edge, leading pair carries 534/1000 of the weight
    t = r["thresholds"]["10mm"]
    assert t["roll_force_N"] == pytest.approx(M * G * 0.534 * 0.75, abs=0.1)
    assert t["tip_force_N"] == pytest.approx(M * G * (466 + 30) / 990, abs=0.1)
    assert t["tips_first"] is False


def test_robot_point_masses_total():
    st, mv = TP.robot_point_masses("UR16e", np.eye(4), np.zeros(6), HOSE_MASSES)
    assert sum(m for m, _ in st + mv) == pytest.approx(33.1 + 4.0)
    # tool mass placed in the flange frame
    Tf = fk("UR16e", np.zeros(6))
    assert np.allclose(mv[-1][1], (Tf @ [-73.15, 0, 0, 1])[:3])


def test_printer_cart_transport_matches_kippen_study():
    """Same numbers as models/printer_robot/kippen.py (frozen in the fixture)."""
    for key, q in (("transport_upright", [0, -90, 0, -90, 0, 0]),
                   ("transport_pose", CART["expected"]["transport_q_deg"])):
        st, mv = TP.robot_point_masses("UR16e", CART["base"], np.radians(q), HOSE_MASSES)
        out = TP.transport(CART["static_masses"] + [(m, p) for m, p in st + mv], CART["castor_axes"],
                           CART["trail_mm"], 1000, CART["wheel_dia_mm"],
                           directions={"-Y": (0, -1), "+Y": (0, 1), "-X": (-1, 0), "+X": (1, 0)})
        for old, new in zip(CART["expected"][key], out["directions"]):
            assert new["impact_speed_m_s"] == pytest.approx(old["kippgeschwindigkeit_anprall_m_s"], abs=0.006)
            assert new["stop_decel_m_s2"] == pytest.approx(old["kippverzoegerung_m_s2"], abs=0.006)
            assert new["push_force_N"] == pytest.approx(old["kippkraft_griff_N"], abs=0.6)
            assert new["cog_to_edge_mm"] == pytest.approx(old["abstand_schwerpunkt_kante_mm"], abs=0.6)
            for e in (5, 10, 15):
                assert new["thresholds"][f"{e}mm"]["tips_first"] == old[f"schwelle_{e}mm"]["kippt_vorher"]


def test_cart_tipping_tool_runs_the_cart_study():
    reach = T.ur_reach_check("UR16e", CART["base"], CART["targets"][::9], HOSE_TOOL, CART["boxes"],
                             spin_step_deg=30)
    out = T.cart_tipping(CART["static_masses"], CART["castor_axes"],
                         robot={"model": "UR16e", "base": CART["base"], "poses_q_deg": reach["poses_q_deg"],
                                "tool_masses": HOSE_MASSES, "tcp": HOSE_TOOL["tcp"],
                                "transport_q_deg": CART["expected"]["transport_q_deg"]},
                         trail_mm=CART["trail_mm"], wheel_dia_mm=CART["wheel_dia_mm"],
                         accelerations_m_s2={"program": 1.5})
    assert out["poses_checked"] == len(reach["poses_q_deg"])
    assert out["total_mass_kg"] == pytest.approx(175.0 + 33.1 + 4.0, abs=0.2)
    assert out["hand_force"]["worst"]["safety_factor"] > 1.5
    assert out["robot_acceleration"]["program"]["safety_factor"] > 4   # study: 5.2 over all 117 poses
    want = {r["kippgeschwindigkeit_anprall_m_s"] for r in CART["expected"]["transport_pose"]}
    assert {r["impact_speed_m_s"] for r in out["transport"]["directions"]} == want
