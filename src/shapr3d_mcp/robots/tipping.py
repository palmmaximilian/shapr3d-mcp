"""Tipping of a free-standing robot cart on castors.

Point masses (kg, xyz mm, floor at z = floor_z) over a support polygon (the
castor swivel axes, or the wheel contacts trailing behind them when rolling).
Per support edge the restoring moment of the weight is compared with:

- static: a horizontal hand force (e.g. pushing the tool while teaching);
- robot motion: every moving robot mass gets the full TCP acceleration,
  horizontal and outwards across the edge (conservative - links near the
  base accelerate less);
- manual transport: pushing force that lifts the cart, sudden stop
  (deceleration), wheel blocked abruptly at speed v (angular momentum about
  the leading contact edge vs. the energy to lift the CoG over it), slope
  angle, and climbing a floor threshold vs. tipping over it.

Swivel castors trail `trail_mm` behind the swivel axis when rolling: the
contact points (and so the tipping edges) shift against the travel direction.
"""

from __future__ import annotations

import numpy as np

from .ur_kinematics import fk_all, get_model

G = 9.81

DIRECTIONS = {"+X": (1, 0), "-X": (-1, 0), "+Y": (0, 1), "-Y": (0, -1)}


def _masses(ms) -> list[tuple[float, np.ndarray]]:
    """[(kg, xyz)] from tuples or {"kg", "xyz"} dicts."""
    out = []
    for m in ms:
        if isinstance(m, dict):
            out.append((float(m["kg"]), np.asarray(m["xyz"], float)))
        else:
            out.append((float(m[0]), np.asarray(m[1], float)))
    return out


def hull_edges(points) -> list[tuple[np.ndarray, np.ndarray]]:
    """Edges of the support polygon, counter-clockwise (inside = left).
    Points are sorted by angle about their mean (convex layouts)."""
    pts = [np.asarray(p, float)[:2] for p in points]
    c = np.mean(pts, axis=0)
    pts = sorted(pts, key=lambda p: np.arctan2(p[1] - c[1], p[0] - c[0]))
    return [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]


def inward(a, b) -> np.ndarray:
    e = (b - a) / np.linalg.norm(b - a)
    return np.array([-e[1], e[0]])


def restoring_moment(masses, a, b) -> float:
    """N m of the weights about edge a-b (> 0 = holds the cart down)."""
    n = inward(a, b)
    return sum(m * G * np.dot(p[:2] - a, n) for m, p in masses) / 1000.0


def _ignored(a, b, ignore_normals) -> bool:
    out = -inward(a, b)
    for n in ignore_normals or ():
        n = np.asarray(n, float)[:2]
        if np.dot(out, n / np.linalg.norm(n)) > 0.9:
            return True
    return False


def _edge(a, b) -> list:
    return [[round(float(v), 1) for v in a], [round(float(v), 1) for v in b]]


def hand_force(masses, support, force_N: float, height_mm: float, floor_z: float = 0.0,
               ignore_edge_normals=None) -> dict:
    """Horizontal force at height_mm pushing outwards across each edge."""
    ms = _masses(masses)
    rows = []
    for a, b in hull_edges(support):
        if _ignored(a, b, ignore_edge_normals):
            continue
        rest = restoring_moment(ms, a, b)
        over = force_N * (height_mm - floor_z) / 1000.0
        rows.append({"edge": _edge(a, b), "restoring_Nm": round(rest, 1),
                     "overturning_Nm": round(over, 1), "net_Nm": round(rest - over, 1),
                     "safety_factor": round(rest / over, 2) if over > 0 else None})
    worst = min(rows, key=lambda r: r["net_Nm"])
    return {"force_N": force_N, "height_mm": height_mm, "worst": worst, "edges": rows}


def acceleration(masses, moving, support, accels: dict, floor_z: float = 0.0,
                 ignore_edge_normals=None) -> dict:
    """Safety factor (restoring / overturning) per acceleration level for
    the moving masses accelerated horizontally at `a` (m/s^2). `masses` are
    all masses (moving ones included) for the restoring moment."""
    ms, mv = _masses(masses), _masses(moving)
    h_m = sum(m * (p[2] - floor_z) for m, p in mv) / 1000.0     # kg m
    out = {}
    for name, acc in accels.items():
        best = None
        for a, b in hull_edges(support):
            if _ignored(a, b, ignore_edge_normals):
                continue
            rest = restoring_moment(ms, a, b)
            over = acc * h_m
            sf = rest / over if over > 0 else float("inf")
            if best is None or sf < best["safety_factor"]:
                best = {"safety_factor": round(sf, 2), "restoring_Nm": round(rest, 1),
                        "overturning_Nm": round(over, 1), "edge": _edge(a, b)}
        out[name] = {"accel_m_s2": acc, **best}
    return out


def transport(masses, castor_axes, trail_mm: float = 0.0, handle_height_mm: float = 1000.0,
              wheel_dia_mm: float | None = None, thresholds_mm=(5, 10, 15),
              directions: dict | None = None, floor_z: float = 0.0) -> dict:
    """Pushing the cart by hand: per travel direction the CoG distance to
    the leading contact edge d, CoG height h and

    - stop_decel_m_s2 = g d / h (deceleration that tips it, e.g. hitting
      something with the cart),
    - push_force_N = M g d / H (horizontal force at handle height H that
      lifts the trailing wheels),
    - impact_speed_m_s: a wheel blocks abruptly at speed v; the cart pivots
      about the leading edge with angular momentum M v h, and tips if the
      kinetic energy lifts the CoG over the edge:
      v = sqrt(2 g dh (k^2 + h^2 + d^2)) / h, dh = sqrt(h^2 + d^2) - h,
      k = radius of gyration about the CoG along the edge,
    - slope_deg = atan(d / h) (standing on a slope, braked),
    - thresholds: push force to roll the two leading wheels onto a step of
      height e vs. the handle force that tips the cart about the step edge.
    """
    ms = _masses(masses)
    M = sum(m for m, _ in ms)
    com = sum(m * p for m, p in ms) / M
    axes = [np.asarray(p, float)[:2] for p in castor_axes]
    res = []
    for name, u in (directions or DIRECTIONS).items():
        u = np.asarray(u, float)
        u /= np.linalg.norm(u)
        contacts = [p - trail_mm * u for p in axes]
        best = None
        for a, b in hull_edges(contacts):
            n = inward(a, b)
            if np.dot(n, u) < -0.5:                       # edge facing the travel direction
                d = float(np.dot(com[:2] - a, n))
                if best is None or d < best[0]:
                    best = (d, a, b)
        d, a, b = best
        h = float(com[2] - floor_z)
        e = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([-e[1], e[0]])
        k2 = sum(m * (np.dot(p[:2] - com[:2], nrm) ** 2 + (p[2] - com[2]) ** 2)
                 for m, p in ms) / M / 1e6
        hm, dm = h / 1000, d / 1000
        dh = np.hypot(hm, dm) - hm
        v = float(np.sqrt(2 * G * dh * (k2 + hm ** 2 + dm ** 2)) / hm)
        row = {"direction": name, "cog_to_edge_mm": round(d, 1), "cog_height_mm": round(h, 1),
               "leading_edge": _edge(a, b),
               "stop_decel_m_s2": round(G * d / h, 2), "stop_decel_g": round(d / h, 3),
               "push_force_N": round(M * G * d / (handle_height_mm - floor_z), 1),
               "impact_speed_m_s": round(v, 2),
               "slope_deg": round(float(np.degrees(np.arctan(d / h))), 1)}
        if wheel_dia_mm:
            row["thresholds"] = _thresholds(M, com, contacts, u, d, wheel_dia_mm / 2,
                                            handle_height_mm - floor_z, thresholds_mm)
        res.append(row)
    return {"mass_kg": round(M, 2), "cog_mm": [round(float(v), 1) for v in com],
            "trail_mm": trail_mm, "handle_height_mm": handle_height_mm, "directions": res}


def _thresholds(M, com, contacts, u, d, R, H, steps) -> dict:
    srt = sorted(contacts, key=lambda p: -np.dot(p, u))
    lead, trail = srt[:2], srt[2:]
    span = float(np.dot(lead[0] - trail[0], u))
    w_lead = M * G * float(np.dot(com[:2] - trail[0], u)) / span   # static share on the leading pair
    out = {}
    for e in steps:
        cos_t = (R - e) / R
        f_roll = w_lead * np.sqrt(1 - cos_t ** 2) / cos_t
        ahead = np.sqrt(R ** 2 - (R - e) ** 2)        # pivot ahead of the wheel contact, e up
        f_tip = M * G * (d + ahead) / (H - e)
        out[f"{e:g}mm"] = {"roll_force_N": round(float(f_roll), 1), "tip_force_N": round(float(f_tip), 1),
                           "tips_first": bool(f_tip < f_roll)}
    return out


def robot_point_masses(model, Tbase, q, tool_masses=()) -> tuple[list, list]:
    """(static, moving) point masses of a UR arm in pose q: base housing
    (static), links 1-6 and tool masses given in the flange frame
    ([(kg, xyz)] or [{"kg", "xyz"}]). Link CoGs: joint origins, upper arm
    and forearm at mid-length on their offset planes (approximate)."""
    m = get_model(model)
    k = m.size
    fr = [np.asarray(Tbase, float) @ f for f in fk_all(m, q)]
    o = [f[:3, 3] for f in fr]
    z1 = fr[1][:3, 2]
    pts = [o[1], (o[1] + o[2]) / 2 + z1 * 176 * k, (o[2] + o[3]) / 2 + z1 * 40 * k, o[4], o[5], o[6]]
    static = [(m.base_mass, o[0] + fr[0][:3, 2] * 60 * k)]
    moving = list(zip(m.link_masses, pts))
    R, t = fr[6][:3, :3], fr[6][:3, 3]
    moving += [(kg, R @ p + t) for kg, p in _masses(tool_masses)]
    return static, moving
