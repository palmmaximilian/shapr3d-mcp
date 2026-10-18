"""MCP tools for UR robot studies: kinematics, reach + base placement,
posing the vendor STEP, real-CAD collision check and cart tipping.

Registered on the server's FastMCP instance when shapr3d_mcp.server is
imported; the heavy ones run in the job worker like the other CAD tools.
Inputs are plain JSON: mm, degrees, kg. Poses are {"xyz": [..],
"rpy_deg": [roll, pitch, yaw]} (fixed axes, R = Rz Ry Rx), {"xyz",
"rotvec": [rad]} (UR pendant) or a 4x4 matrix.
"""

from __future__ import annotations

import numpy as np

from shapr3d_mcp.server import _export, _heavy_tool, _import_cached, _out_base, _resolve, mcp

from . import reach as R
from . import tipping as TP
from . import ur_pose as UP
from .ur_kinematics import MODELS, fk_all, get_model, ik, pose, pose_dict

MAX_BASES = 50_000


def _deg(q) -> list[float]:
    return [round(float(v), 3) for v in np.degrees(q)]


def _q(q_deg) -> np.ndarray:
    q = np.radians(np.asarray(q_deg, float))
    if q.shape != (6,):
        raise ValueError(f"q_deg needs 6 joint angles, got {list(q_deg)}")
    return q


def _poses_q(q_deg) -> list[np.ndarray]:
    """One 6-vector or a list of them."""
    a = np.asarray(q_deg, float)
    return [_q(a)] if a.ndim == 1 else [_q(r) for r in a]


def _file(path: str):
    src = _resolve(path)
    if not src.exists():
        raise FileNotFoundError(f"No such file: {src}")
    return src


@mcp.tool()
def ur_kinematics(model: str, q_deg: list[float] | None = None,
                  tcp_pose: dict | list | None = None, base: dict | list | None = None,
                  tcp: dict | list | None = None) -> dict:
    """UR forward / inverse kinematics (UR3e, UR5e, UR10e, UR12e, UR16e,
    UR20; official DH values).

    Args:
        model: robot, e.g. "UR16e".
        q_deg: joint angles (deg) -> forward kinematics: flange and TCP
            pose plus every DH frame origin, in the world.
        tcp_pose: target TCP pose in the world -> all IK solutions (up to 8,
            deg in [-180, 180), FK-verified). Omit both for the model data.
        base: robot base pose in the world ({"xyz", "yaw_deg", "lean_deg",
            "lean_toward"} or a pose); default = world.
        tcp: TCP frame in the flange frame (tool offset); default = flange.
    """
    m = get_model(model)
    Tb, Tt = R.base_from_spec(base), pose(tcp)
    out = {"model": m.as_dict(), "base": pose_dict(Tb)}
    if q_deg is not None:
        fr = [Tb @ f for f in fk_all(m, _q(q_deg))]
        out.update(q_deg=list(q_deg), flange=pose_dict(fr[-1]), tcp=pose_dict(fr[-1] @ Tt),
                   joint_origins_mm=[[round(float(v), 2) for v in f[:3, 3]] for f in fr])
    if tcp_pose is not None:
        Tf = np.linalg.inv(Tb) @ pose(tcp_pose) @ np.linalg.inv(Tt)
        sols = ik(m, Tf)
        out.update(tcp_target=pose_dict(pose(tcp_pose)), solutions_q_deg=[_deg(q) for q in sols])
        if not sols:
            out["note"] = "unreachable (or wrist-singular: q5 = 0)"
    if q_deg is None and tcp_pose is None:
        out["models"] = sorted(MODELS)
    return out


def _reach_rows(m, Tb, targets, tool, lo, hi, spin, tilts, margin, limits):
    ok_fn = R.joint_check(limits)
    rows = []
    for t in targets:
        b = R.reach_target(m, Tb, t, tool, lo, hi, spin, tilts, margin, ok_fn)
        row = {"name": t["name"], "xyz": t["xyz"], "reachable": b is not None}
        if b is not None:
            c, q, az, tl = b
            clear, ci, bi = R.pose_clearance(m, Tb, q, tool, lo, hi, detail=True)
            names = list(R.CAPSULE_NAMES) + [f"tool capsule {i}" for i in range(len(tool.capsules))]
            row.update(clearance_mm=None if np.isinf(c) else round(float(c), 1),
                       q_deg=_deg(q), spin_deg=round(float(np.degrees(az)), 1),
                       tilt_deg=round(float(np.degrees(tl)), 1))
            if ci is not None:
                row["closest"] = [names[ci], bi]
        rows.append(row)
    return rows


@_heavy_tool
def ur_reach_check(model: str, base: dict | list, targets: list,
                   tool: dict | None = None, obstacles: list | dict | None = None,
                   spin_step_deg: float = 15.0, tilts_deg: list[float] | None = None,
                   margin_mm: float = 4.0, joint_limits_deg: list | None = None) -> dict:
    """Can a UR robot reach every target collision-free? Analytic IK (all 8
    solutions) x sampled tool spins/tilts; links and tool are capsules,
    obstacles are boxes. Picks the pose with the largest clearance per
    target. Confirm the chosen poses with ur_collision_check (real CAD).

    Args:
        model: e.g. "UR16e".
        base: robot base in the world: {"xyz": [mm], "yaw_deg": -90
            (cable outlet = base +y -> world +X), "lean_deg", "lean_toward":
            [x, y] | "lean_dir_deg"} or a pose / 4x4.
        targets: TCP points [[x, y, z], ...] or dicts {"name", "xyz",
            "approach": [0, 0, -1] (TCP z direction, default straight down),
            "spin_deg" (fixed; default free -> sampled), "tilts_deg"} or full
            TCP poses {"name", "xyz", "rpy_deg"}.
        tool: {"tcp": pose of the TCP in the flange frame, "capsules":
            [{"p0": [..], "p1": [..], "r": mm}] in the flange frame}.
            E.g. a nozzle 410 mm out and 73 mm off-axis:
            {"tcp": {"xyz": [-73.15, 0, 410]}}.
        obstacles: boxes [{"name", "min": [..], "max": [..]}] or
            {name: [[min], [max]]} in the world frame.
        spin_step_deg: sampling of the free rotation about the approach.
        tilts_deg: tilt angles tried per target (default [0]).
        margin_mm: required capsule clearance.
        joint_limits_deg: 6 [lo, hi] (null = free); default = keep the
            upper arm out of the base and the elbow < 170 deg folded.
    Returns per target: reachable, clearance_mm, q_deg, spin/tilt and the
    closest capsule/box; plus poses_q_deg (reached ones) for the CAD check.
    """
    m = get_model(model)
    Tb = R.base_from_spec(base)
    tl = R.Tool.from_spec(tool)
    lo, hi, names = R.boxes(obstacles)
    tg = R.targets_from_spec(targets)
    rows = _reach_rows(m, Tb, tg, tl, lo, hi, spin_step_deg, tuple(tilts_deg or [0.0]),
                       margin_mm, joint_limits_deg)
    for r in rows:
        if "closest" in r and r["closest"][1] is not None:
            r["closest"][1] = names[r["closest"][1]]
    ok = [r for r in rows if r["reachable"]]
    clear = [r["clearance_mm"] for r in ok if r["clearance_mm"] is not None]
    return {"model": m.name, "base": pose_dict(Tb), "targets": len(rows), "reached": len(ok),
            "all_reached": len(ok) == len(rows),
            "min_clearance_mm": min(clear) if clear else None,
            "unreached": [r["name"] for r in rows if not r["reachable"]],
            "results": rows, "poses_q_deg": [r["q_deg"] for r in ok]}


@_heavy_tool
def ur_base_search(model: str, targets: list, grid: dict, tool: dict | None = None,
                   obstacles: list | dict | None = None, spin_step_deg: float = 30.0,
                   tilts_deg: list[float] | None = None, margin_mm: float = 4.0,
                   joint_limits_deg: list | None = None, top: int = 15) -> dict:
    """Grid search for robot base placements that reach every target
    collision-free (same model and inputs as ur_reach_check), in parallel.

    Args:
        grid: {"x", "y", "z", optional "lean_deg", "yaw_deg" (default 0),
            "lean_dir_deg"}: each a number, a list or {"from", "to",
            "step"}; "lean_toward": [x, y] for leaning bases.
            E.g. {"x": {"from": 1000, "to": 1300, "step": 25}, "y": [600],
            "z": {"from": 1300, "to": 1900, "step": 50}, "lean_deg": [0, 30],
            "yaw_deg": -90, "lean_toward": [655, 1185]}.
        top: how many of the best bases to return.
    Returns the best bases (all targets reached, largest worst-case
    clearance first) and counts. Refine the winner with ur_reach_check.
    """
    m = get_model(model)
    n = len(R.base_grid(grid))
    if n > MAX_BASES:
        raise ValueError(f"grid has {n} bases (max {MAX_BASES}); coarsen it and refine")
    lo, hi, _ = R.boxes(obstacles)
    rows = R.grid_search(m, R.targets_from_spec(targets), grid, R.Tool.from_spec(tool), lo, hi,
                         spin_step_deg, tuple(tilts_deg or [0.0]), margin_mm, joint_limits_deg)
    good = [r for r in rows if r["ok"]]
    return {"model": m.name, "bases_checked": n, "bases_ok": len(good),
            "targets": len(targets), "best": good[:top],
            "closest_misses": [] if good else rows[:top]}


def _tool_part(tool_step: str | None, tool_mount, Tf):
    if not tool_step:
        return None
    t = _import_cached(_file(tool_step))
    part = t.moved(UP.to_loc(Tf @ pose(tool_mount)))
    part.label = "tool"
    return part


@_heavy_tool
def ur_pose_step(model: str, robot_step: str, q_deg: list[float],
                 base: dict | list | None = None, tool_step: str | None = None,
                 tool_mount: dict | list | None = None, link_map: dict[str, int] | None = None,
                 output_name: str | None = None, formats: list[str] | None = None) -> dict:
    """Pose the vendor UR STEP (universal-robots.com, arm straight up, Y up)
    at joint angles q_deg and export the posed robot (+ tool) as an
    assembly.

    Args:
        robot_step: vendor STEP of the arm; each top-level child is one link.
        q_deg: 6 joint angles (deg).
        base: robot base pose in the world (see ur_reach_check).
        tool_step: optional tool/end-effector file to mount on the flange.
        tool_mount: pose of the tool file's frame in the flange frame (flange
            z points out of the face), e.g. {"rpy_deg": [180, 0, 0]}.
        link_map: {child label: DH frame 0..6}; default = detected from the
            home pose (base, shoulder, upper arm, forearm, wrist 1-3).
        output_name: workspace name (default "<model>_posed").
        formats: default ["step"].
    """
    m = get_model(model)
    Tb = R.base_from_spec(base)
    q = _q(q_deg)
    from build123d import Compound

    robot = _import_cached(_file(robot_step))
    parts = UP.posed_robot(robot, m, q, Tb, link_map)
    Tf = Tb @ fk_all(m, q)[-1]
    tool = _tool_part(tool_step, tool_mount, Tf)
    asm = Compound(label=f"{m.name} posed", children=parts + ([tool] if tool else []))
    files = _export(asm, _out_base(output_name or f"{m.name}_posed"), formats or ["step"])
    return {"files": files, "model": m.name, "q_deg": list(q_deg), "flange": pose_dict(Tf),
            "parts": [p.label for p in asm.children]}


def _volume(a, b) -> float:
    inter = a & b
    return float(sum(s.volume for s in inter.solids())) if inter else 0.0


@_heavy_tool
def ur_collision_check(model: str, robot_step: str, poses_q_deg: list,
                       environment: list[str], base: dict | list | None = None,
                       tool_step: str | None = None, tool_mount: dict | list | None = None,
                       link_map: dict[str, int] | None = None, include_base: bool = False,
                       min_volume_mm3: float = 1.0, export_name: str | None = None) -> dict:
    """Real-CAD interference check: the vendor STEP posed per joint vector
    (+ tool) against environment STEP files (exact booleans, slow for big
    files - check the tightest poses from ur_reach_check).

    Args:
        poses_q_deg: one 6-vector or a list of them (deg).
        environment: STEP/BREP files (printer, cart, fixtures) in the same
            world frame as base.
        include_base: also check the fixed base link (off: it usually sits on
            the cart plate on purpose).
        min_volume_mm3: smaller overlaps count as touching, not clashing.
        export_name: also export each posed robot + tool as
            "<export_name>_<i>" STEP.
    Returns per pose and environment file: interference volume per robot
    part (only parts above min_volume_mm3) and a clash flag.
    """
    m = get_model(model)
    Tb = R.base_from_spec(base)
    from build123d import Compound

    robot = _import_cached(_file(robot_step))
    envs = [(p, _import_cached(_file(p))) for p in environment]
    out = []
    for i, q in enumerate(_poses_q(poses_q_deg)):
        parts = UP.posed_robot(robot, m, q, Tb, link_map)
        Tf = Tb @ fk_all(m, q)[-1]
        tool = _tool_part(tool_step, tool_mount, Tf)
        moving = (parts if include_base else parts[1:]) + ([tool] if tool else [])
        row = {"q_deg": _deg(q), "flange": pose_dict(Tf), "clash": False, "environment": {}}
        for path, env in envs:
            vols = {p.label: round(_volume(p, env), 1) for p in moving}
            hits = {k: v for k, v in vols.items() if v >= min_volume_mm3}
            row["environment"][path] = {"interference_mm3": round(sum(hits.values()), 1), "parts": hits}
            row["clash"] |= bool(hits)
        if export_name:
            row["files"] = _export(Compound(children=parts + ([tool] if tool else [])),
                                   _out_base(f"{export_name}_{i}"), ["step"])
        out.append(row)
    return {"model": m.name, "poses": len(out), "clashing_poses": sum(r["clash"] for r in out),
            "results": out}


def _mass_list(masses) -> list[tuple[float, list]]:
    """[{"name", "kg", "xyz"}] or mass_properties-style {"mass_kg", "cog_mm"}."""
    out = []
    for mm in masses:
        if isinstance(mm, dict):
            kg = mm.get("kg", mm.get("mass_kg"))
            xyz = mm.get("xyz", mm.get("cog_mm"))
            if kg is None or xyz is None:
                raise ValueError(f"mass entry needs kg + xyz: {mm}")
            out.append((float(kg), list(xyz)))
        else:
            out.append((float(mm[0]), list(mm[1])))
    return out


@_heavy_tool
def cart_tipping(masses: list, castors: list[list[float]], robot: dict | None = None,
                 trail_mm: float = 0.0, wheel_dia_mm: float | None = None,
                 handle_height_mm: float = 1000.0, hand_force_N: float = 150.0,
                 hand_force_height_mm: float | None = None,
                 accelerations_m_s2: dict[str, float] | None = None,
                 ignore_edge_normals: list[list[float]] | None = None,
                 thresholds_mm: list[float] | None = None, floor_z: float = 0.0) -> dict:
    """Tipping safety of a robot cart on castors: hand force on the tool,
    robot accelerations (program / emergency stops) and manual transport
    (push force, sudden stop, wheel blocked at speed, slope, thresholds).

    Args:
        masses: static cart masses [{"name", "kg", "xyz": [mm]}] (or
            mass_properties parts with mass_kg / cog_mm), world frame, floor
            at z = floor_z. Do not include the robot if `robot` is given.
        castors: swivel axis positions [[x, y], ...] (support polygon).
        robot: {"model", "base" (pose/spec), "poses_q_deg": [[..], ..] (the
            working poses, e.g. ur_reach_check's poses_q_deg),
            "tool_masses": [{"kg", "xyz"}] in the flange frame (tool, hose
            share...), "tcp": TCP in the flange frame (where the hand force
            acts), "transport_q_deg": pose while pushed (default: the first
            working pose)}.
        trail_mm: castor trail (contact behind the swivel axis when rolling).
        wheel_dia_mm: enables the threshold check.
        handle_height_mm: where a person pushes the cart.
        hand_force_N: horizontal force at the TCP (or hand_force_height_mm).
        accelerations_m_s2: {"label": a} for the moving robot masses; default
            {"program 1.5 m/s2": 1.5, "stop 2.5 m/s2": 2.5, "stop 5 m/s2": 5}.
        ignore_edge_normals: outward normals [x, y] of edges that cannot tip
            (e.g. [[0, 1]] when the cart stands against a machine at +Y).
        thresholds_mm: step heights (default [5, 10, 15]).
    Safety factor = restoring / overturning moment (> 1 holds; aim >= 1.5).
    """
    stat = _mass_list(masses)
    accels = accelerations_m_s2 or {"program 1.5 m/s2": 1.5, "stop 2.5 m/s2": 2.5, "stop 5 m/s2": 5.0}
    poses: list = [None]
    if robot:
        m = get_model(robot["model"])
        Tb = R.base_from_spec(robot.get("base"))
        Tt = pose(robot.get("tcp"))
        poses = _poses_q(robot["poses_q_deg"]) if robot.get("poses_q_deg") is not None else [None]
        if poses[0] is None:
            raise ValueError("robot needs poses_q_deg")
    hand, accel = None, {}
    for q in poses:
        if q is None:
            allm, moving = stat, []
            h = hand_force_height_mm if hand_force_height_mm is not None else handle_height_mm
        else:
            st, mv = TP.robot_point_masses(m, Tb, q, _mass_list(robot.get("tool_masses", [])))
            allm, moving = stat + st + mv, mv
            tcp_z = float((Tb @ fk_all(m, q)[-1] @ Tt)[2, 3])
            h = hand_force_height_mm if hand_force_height_mm is not None else tcp_z
        hf = TP.hand_force(allm, castors, hand_force_N, h, floor_z, ignore_edge_normals)
        if hand is None or hf["worst"]["net_Nm"] < hand["worst"]["net_Nm"]:
            hand = {**hf, "pose_q_deg": None if q is None else _deg(q)}
        if moving:
            for k, v in TP.acceleration(allm, moving, castors, accels, floor_z, ignore_edge_normals).items():
                if k not in accel or v["safety_factor"] < accel[k]["safety_factor"]:
                    accel[k] = {**v, "pose_q_deg": _deg(q)}
    hand.pop("edges")
    tq = None
    allm = stat
    if robot:
        tq = _q(robot["transport_q_deg"]) if robot.get("transport_q_deg") is not None else poses[0]
        st, mv = TP.robot_point_masses(m, Tb, tq, _mass_list(robot.get("tool_masses", [])))
        allm = stat + st + mv
    tr = TP.transport(allm, castors, trail_mm, handle_height_mm, wheel_dia_mm,
                      thresholds_mm if thresholds_mm is not None else (5, 10, 15), floor_z=floor_z)
    tr["pose_q_deg"] = None if tq is None else _deg(tq)
    return {"total_mass_kg": tr["mass_kg"], "hand_force": hand,
            "robot_acceleration": accel or None, "transport": tr,
            "poses_checked": 0 if poses == [None] else len(poses)}
