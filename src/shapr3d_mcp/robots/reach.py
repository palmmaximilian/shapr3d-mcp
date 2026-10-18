"""Reach + collision study for a UR robot with a tool among box obstacles.

Robot links (and the tool) are capsules in the DH frames, the environment is
axis-aligned boxes (e.g. read off section cuts). For each TCP target all 8 IK
solutions are tried over sampled tool spins (rotation about the approach axis)
and tilts; the pose with the largest clearance wins. Re-check the chosen poses
against the real CAD (ur_pose + a boolean, see the ur_collision_check tool).

World frame: whatever the boxes and targets use (mm). Tbase = world <- robot
base (z up through joint 1, cable outlet towards base +y).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .ur_kinematics import URModel, axis_angle, fk_all, get_model, ik, pose

# ---- tool ------------------------------------------------------------------


@dataclass
class Tool:
    """tcp: TCP frame in the flange frame (4x4). capsules: (p0, p1, radius)
    in the flange frame, checked against the obstacles."""
    tcp: np.ndarray = field(default_factory=lambda: np.eye(4))
    capsules: list = field(default_factory=list)

    @classmethod
    def from_spec(cls, spec: dict | None) -> "Tool":
        """{"tcp": {"xyz": [mm], "rpy_deg": [...]}, "capsules": [{"p0": [..],
        "p1": [..], "r": mm}, ...]} (all in the flange frame) or None."""
        if spec is None:
            return cls()
        if isinstance(spec, Tool):
            return spec
        caps = [(np.asarray(c["p0"], float), np.asarray(c["p1"], float), float(c["r"]))
                for c in spec.get("capsules", [])]
        return cls(pose(spec.get("tcp")), caps)


# ---- obstacles -------------------------------------------------------------


def boxes(spec) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(lo N x 3, hi N x 3, names) from [{"min": [..], "max": [..], "name"}]
    or {name: [[min], [max]]} or [[min], [max]] pairs."""
    if spec is None or len(spec) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), []
    items = spec.items() if isinstance(spec, dict) else enumerate(spec)
    lo, hi, names = [], [], []
    for k, b in items:
        if isinstance(b, dict):
            a, c, k = b["min"], b["max"], b.get("name", k)
        else:
            a, c = b
        a, c = np.asarray(a, float), np.asarray(c, float)
        lo.append(np.minimum(a, c))
        hi.append(np.maximum(a, c))
        names.append(str(k))
    return np.array(lo), np.array(hi), names


# ---- capsule model ---------------------------------------------------------


def link_capsules(model: str | URModel, frames) -> list:
    """(p0, p1, r) per link housing from DH frames (any common frame).
    Measured on the UR16e vendor STEP; the UR10e/UR12e share its housings,
    other models are scaled with d4 (approximate)."""
    m = get_model(model)
    k = m.size
    o = [f[:3, 3] for f in frames]
    z = [f[:3, 2] for f in frames]
    zb, yb = frames[0][:3, 2], frames[0][:3, 1]
    s_up, s_fo = 176.0 * k, 40.0 * k            # upper arm / forearm offsets along z1
    return [
        (o[0], o[0] + zb * (m.d1 - 40 * k), 100.0 * k),                   # base
        (o[1] - 10 * k * z[1], o[1] + 230 * k * z[1], 80.0 * k),          # shoulder
        (o[1] + s_up * z[1], o[2] + s_up * z[1], 65.0 * k),               # upper arm
        (o[2] + s_fo * z[1], o[2] + 230 * k * z[1], 75.0 * k),            # elbow
        (o[2] + s_fo * z[1], o[3] + s_fo * z[1], 55.0 * k),               # forearm
        (o[3] + s_fo * z[1], o[4] + 50 * k * z[3], 52.0 * k),             # wrist 1
        (o[4] - 55 * k * z[4], o[5], 52.0 * k),                           # wrist 2
        (o[5] - 60 * k * z[5], o[6] - 50 * k * z[5], 50.0 * k),           # wrist 3
        (o[0] + 40 * k * zb + 100 * k * yb, o[0] + 40 * k * zb + 300 * k * yb, 30.0 * k),  # cable
    ]


def tool_capsules(tool: Tool, Tf) -> list:
    R, t = Tf[:3, :3], Tf[:3, 3]
    return [(R @ a + t, R @ b + t, r) for a, b, r in tool.capsules]


def seg_box_clear(p0, p1, r, lo, hi, n: int = 12) -> float:
    """Clearance (mm) of a capsule to the nearest box, sampled at n points
    along its axis; negative = penetration (approximate)."""
    if len(lo) == 0:
        return float("inf")
    t = np.linspace(0, 1, n)[:, None]
    pts = p0 + t * (p1 - p0)
    d = np.maximum(np.maximum(lo[None] - pts[:, None], pts[:, None] - hi[None]), 0)
    return float(np.linalg.norm(d, axis=2).min() - r)


def capsules(model, Tbase, q, tool: Tool | None = None) -> list:
    fr = [Tbase @ f for f in fk_all(model, q)]
    caps = link_capsules(model, fr)
    if tool is not None:
        caps += tool_capsules(tool, fr[-1])
    return caps


def pose_clearance(model, Tbase, q, tool: Tool | None, lo, hi, detail: bool = False):
    """Smallest capsule clearance (mm) of robot + tool to the boxes. With
    detail=True also (capsule index, box index) of that minimum."""
    caps = capsules(model, Tbase, q, tool)
    if not detail:
        return min(seg_box_clear(a, b, r, lo, hi) for a, b, r in caps)
    best = (float("inf"), None, None)
    for ci, (a, b, r) in enumerate(caps):
        for bi in range(len(lo)):
            c = seg_box_clear(a, b, r, lo[bi:bi + 1], hi[bi:bi + 1])
            if c < best[0]:
                best = (c, ci, bi)
    return best


CAPSULE_NAMES = ("base", "shoulder", "upper arm", "elbow", "forearm", "wrist 1",
                 "wrist 2", "wrist 3", "base cable")


# ---- joint limits ----------------------------------------------------------


def default_joint_ok(q) -> bool:
    """Self-collision heuristic: upper arm not folded into the base,
    elbow not folded flat (|q2 + 90| < 170, |q3| < 170 deg)."""
    return abs(q[1] + np.pi / 2) < np.radians(170) and abs(q[2]) < np.radians(170)


def joint_check(limits_deg=None):
    """Callable q -> bool: None = default_joint_ok, else 6 [lo, hi] deg
    applied to the wrapped angles (null entries = free)."""
    if limits_deg is None:
        return default_joint_ok
    lim = [None if l is None else np.radians(l) for l in limits_deg]

    def ok(q):
        return all(l is None or l[0] <= v <= l[1] for v, l in zip(q, lim))
    return ok


# ---- targets ---------------------------------------------------------------


def base_T(xyz, yaw_deg: float = -90.0, lean_deg: float = 0.0, lean_toward=None,
           lean_dir_deg: float | None = None) -> np.ndarray:
    """Robot base pose in the world. yaw_deg turns the base about world Z
    (-90: cable outlet, base +y, points to world +X). lean_deg tilts base Z
    towards a world XY point (lean_toward) or a direction (lean_dir_deg,
    from +X)."""
    T = np.eye(4)
    T[:3, 3] = xyz
    Rz = axis_angle((0, 0, 1), np.radians(yaw_deg))
    T[:3, :3] = Rz
    if lean_deg:
        if lean_dir_deg is not None:
            d = np.array([np.cos(np.radians(lean_dir_deg)), np.sin(np.radians(lean_dir_deg))])
        elif lean_toward is not None:
            d = np.asarray(lean_toward[:2], float) - np.asarray(xyz[:2], float)
            d /= np.linalg.norm(d)
        else:
            raise ValueError("lean_deg needs lean_toward [x, y] or lean_dir_deg")
        ax = np.array([-d[1], d[0], 0.0])
        T[:3, :3] = axis_angle(ax, np.radians(lean_deg)) @ Rz
    return T


def base_from_spec(spec) -> np.ndarray:
    """{"xyz", "yaw_deg", "lean_deg", "lean_toward" | "lean_dir_deg"}, a
    pose spec ({"xyz", "rpy_deg"} / 4x4) or None (= identity)."""
    if spec is None:
        return np.eye(4)
    if isinstance(spec, dict) and any(k in spec for k in ("yaw_deg", "lean_deg")):
        return base_T(spec.get("xyz", (0, 0, 0)), spec.get("yaw_deg", 0.0),
                      spec.get("lean_deg", 0.0), spec.get("lean_toward"), spec.get("lean_dir_deg"))
    return pose(spec)


def _spin_frame(z, az):
    z = np.asarray(z, float) / np.linalg.norm(z)
    ref = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = ref - ref @ z * z
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, z)
    return z, np.cos(az) * e1 + np.sin(az) * e2


def tcp_pose(xyz, approach=(0, 0, -1), spin: float = 0.0, tilt: float = 0.0) -> np.ndarray:
    """TCP pose (world): z along `approach`, x at angle `spin` (rad) around
    it, then z and x tilted by `tilt` about z x x."""
    z, x = _spin_frame(approach, spin)
    if tilt:
        ax = np.cross(z, x)
        R = axis_angle(ax, tilt)
        z, x = R @ z, R @ x
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x, np.cross(z, x), z
    T[:3, 3] = xyz
    return T


def target_poses(target: dict, spin_step_deg: float = 15.0, tilts_deg=(0.0,)):
    """[(TCP pose, spin rad, tilt rad)] for one target: {"xyz", "approach"
    (default [0,0,-1]), "spin_deg" (fixed; default free = sampled),
    "tilts_deg"} or a full TCP pose {"xyz", "rpy_deg"} / {"xyz", "rotvec"}."""
    if any(k in target for k in ("rpy_deg", "rotvec", "matrix")):
        return [(pose(target), 0.0, 0.0)]
    tilts = np.radians(target.get("tilts_deg", tilts_deg))
    if target.get("spin_deg") is not None:
        spins = [np.radians(target["spin_deg"])]
    else:
        spins = np.radians(np.arange(0, 360, spin_step_deg))
    approach = target.get("approach", (0, 0, -1))
    return [(tcp_pose(target["xyz"], approach, az, tl), az, tl) for tl in tilts for az in spins]


def targets_from_spec(spec) -> list[dict]:
    """[[x,y,z], ...] or [{"xyz", ...}, ...] -> list of target dicts."""
    out = []
    for i, t in enumerate(spec):
        t = dict(t) if isinstance(t, dict) else {"xyz": list(t)}
        t.setdefault("name", f"t{i}")
        out.append(t)
    return out


def box_grid_targets(lo, hi, n, inset: float = 0.0, name: str = "grid") -> list[dict]:
    """n = (nx, ny, nz) evenly spaced targets in a box, inset from its faces;
    a count of 1 puts the row at the centre of that axis."""
    axes = []
    for a, b, k in zip(lo, hi, n):
        a, b = a + inset, b - inset
        axes.append([(a + b) / 2] if k == 1 else np.linspace(a, b, k))
    return [{"name": name, "xyz": [float(x), float(y), float(z)]}
            for x in axes[0] for y in axes[1] for z in axes[2]]


# ---- reach -----------------------------------------------------------------


def reach_target(model, Tbase, target: dict, tool: Tool | None = None, lo=None, hi=None,
                 spin_step_deg: float = 15.0, tilts_deg=(0.0,), margin: float = 4.0,
                 joint_ok=default_joint_ok):
    """Best collision-free IK solution: (clearance, q, spin, tilt) or None."""
    tool = tool or Tool()
    lo = np.zeros((0, 3)) if lo is None else lo
    hi = np.zeros((0, 3)) if hi is None else hi
    Tinv = np.linalg.inv(Tbase)
    tcp_inv = np.linalg.inv(tool.tcp)
    best = None
    for Ttcp, az, tl in target_poses(target, spin_step_deg, tilts_deg):
        Tf = Ttcp @ tcp_inv
        for q in ik(model, Tinv @ Tf):
            if not joint_ok(q):
                continue
            c = pose_clearance(model, Tbase, q, tool, lo, hi)
            if c > margin and (best is None or c > best[0]):
                best = (c, q, az, tl)
    return best


def evaluate_base(model, Tbase, targets, tool=None, lo=None, hi=None, spin_step_deg=15.0,
                  tilts_deg=(0.0,), margin=4.0, joint_ok=default_joint_ok, stop_on_fail=True):
    """(all reached, [(target, best | None)]); farthest targets first so a
    failing base is rejected early."""
    o = Tbase[:3, 3]
    order = sorted(targets, key=lambda t: -np.hypot(t["xyz"][0] - o[0], t["xyz"][1] - o[1]))
    res = []
    for t in order:
        b = reach_target(model, Tbase, t, tool, lo, hi, spin_step_deg, tilts_deg, margin, joint_ok)
        res.append((t, b))
        if b is None and stop_on_fail:
            return False, res
    return all(b is not None for _, b in res), res


def _grid_axis(v):
    if isinstance(v, (int, float)):
        return [float(v)]
    if isinstance(v, dict):
        return list(np.arange(v["from"], v["to"] + 1e-9, v["step"]))
    return [float(x) for x in v]


def base_grid(spec: dict) -> list[dict]:
    """Base specs from {"x", "y", "z", "lean_deg", "yaw_deg" (default 0), "lean_dir_deg"}:
    each a number, a list, or {"from", "to", "step"}. "lean_toward" [x, y]
    is shared by all."""
    keys = ["x", "y", "z", "yaw_deg", "lean_deg", "lean_dir_deg"]
    axes = {k: _grid_axis(spec[k]) for k in keys if k in spec}
    for k in ("x", "y", "z"):
        if k not in axes:
            raise ValueError(f"base grid needs {k!r}")
    out = [{}]
    for k, vals in axes.items():
        out = [{**o, k: v} for o in out for v in vals]
    bases = []
    for o in out:
        b = {"xyz": [o["x"], o["y"], o["z"]], "yaw_deg": o.get("yaw_deg", spec.get("default_yaw_deg", 0.0)),
             "lean_deg": o.get("lean_deg", 0.0)}
        if "lean_dir_deg" in o:
            b["lean_dir_deg"] = o["lean_dir_deg"]
        elif spec.get("lean_toward") is not None:
            b["lean_toward"] = spec["lean_toward"]
        bases.append(b)
    return bases


def _grid_job(args):
    model, base, targets, tool, lo, hi, spin, tilts, margin, limits = args
    ok, res = evaluate_base(model, base_from_spec(base), targets, tool, lo, hi, spin, tilts,
                            margin, joint_check(limits), stop_on_fail=True)
    clear = min((b[0] for _, b in res if b is not None), default=None)
    return {"base": base, "ok": ok, "reached": sum(b is not None for _, b in res),
            "min_clearance_mm": None if clear is None or not ok else round(float(clear), 1)}


def grid_search(model, targets, grid: dict, tool=None, lo=None, hi=None, spin_step_deg=30.0,
                tilts_deg=(0.0,), margin=4.0, joint_limits_deg=None, workers: int | None = None):
    """Evaluate every base in base_grid(grid); returns rows sorted with the
    bases reaching every target first, largest worst-case clearance first."""
    m = get_model(model).name
    tool = Tool.from_spec(tool) if not isinstance(tool, Tool) else tool
    jobs = [(m, b, targets, tool, lo, hi, spin_step_deg, tilts_deg, margin, joint_limits_deg)
            for b in base_grid(grid)]
    workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    rows = None
    if workers > 1 and len(jobs) > 8:
        try:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as ex:
                rows = list(ex.map(_grid_job, jobs, chunksize=max(1, len(jobs) // (workers * 8))))
        except (OSError, RuntimeError, AssertionError):
            rows = None  # e.g. no subprocesses allowed here: run serially
    if rows is None:
        rows = [_grid_job(j) for j in jobs]
    rows.sort(key=lambda r: (not r["ok"], -(r["min_clearance_mm"] or -1e9), -r["reached"]))
    return rows
