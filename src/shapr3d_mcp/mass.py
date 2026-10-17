"""Mass, centre of gravity and inertia of labelled assemblies.

Each top-level assembly child (a "part") gets a material from a
label-substring mapping, or a catalogue mass for vendor parts whose CAD
is hollow-less or simplified (density is scaled so the part weighs what
the datasheet says; the CoG still comes from the geometry, i.e. uniform
density). Everything is reported in the file's world frame - for our
end-effector models that is the robot tool flange.

Units: mass kg, lengths mm, inertia kg*m^2.
"""

from __future__ import annotations

import bisect

import numpy as np

from shapr3d_mcp import kernel

# g/cm^3. Print-process values are for the printed part, not the powder.
DENSITIES = {
    "pa12": 0.95,            # SLS PA12 (PA 2200 parts ~0.93-0.95)
    "pa12 mjf": 1.01,        # HP MJF PA12
    "pa6": 1.14,
    "nylon": 1.14,
    "pla": 1.24,
    "petg": 1.27,
    "abs": 1.04,
    "pom": 1.41,
    "aluminium": 2.70,
    "aluminum": 2.70,
    "steel": 7.85,
    "stainless": 7.93,
    "brass": 8.50,
    "titanium": 4.43,
}

# Rated payload vs CoG distance from the tool flange centre, digitised from
# the UR10e technical specification (Sept 2026) payload curve: (mm, kg).
# Valid for regular mounting; offsets beyond the last point are off-chart.
PAYLOAD_CURVES = {
    "ur10e": [(0, 12.5), (112, 12.5), (150, 12.0), (200, 11.2),
              (250, 10.55), (300, 10.0), (350, 9.6)],
}


def density(material) -> float:
    """g/cm^3 from a number or a DENSITIES name (case/punctuation-free)."""
    if isinstance(material, (int, float)):
        if material <= 0:
            raise ValueError(f"density must be > 0, got {material}")
        return float(material)
    key = kernel.norm(str(material))
    if key in DENSITIES:
        return DENSITIES[key]
    try:
        value = float(str(material))
    except ValueError:
        value = None
    if value is not None:
        return density(value)
    raise ValueError(f"Unknown material {material!r}; use a density in g/cm3 "
                     f"or one of {sorted(DENSITIES)}")


def _match(label: str, mapping: dict | None):
    """Value of the longest key of `mapping` contained in `label`
    (normalised); '*' is the fallback. None when nothing matches."""
    if not mapping:
        return None
    name = kernel.norm(label)
    hits = [k for k in mapping if k != "*" and kernel.norm(k) and kernel.norm(k) in name]
    if hits:
        return mapping[max(hits, key=lambda k: len(kernel.norm(k)))]
    return mapping.get("*")


def _placed(solid, measures: kernel.Measures):
    """(volume mm^3, centroid mm, inertia about centroid mm^5) of a placed
    solid at unit density, from the per-TShape memo."""
    vol, c, inertia = measures.gprops(solid)
    trsf = solid.wrapped.Location().Transformation()
    rot = np.array([[trsf.Value(r, k) for k in (1, 2, 3)] for r in (1, 2, 3)])
    off = np.array([trsf.Value(r, 4) for r in (1, 2, 3)])
    scale = abs(np.linalg.det(rot)) ** (1 / 3)
    if abs(scale - 1) > 1e-9:
        # scaled placement: memo holds the unscaled solid
        vol, c, inertia = vol * scale ** 3, c * scale, inertia * scale ** 5
        rot = rot / scale
    if vol < 0:  # reversed solid
        vol, inertia = -vol, -inertia
    return vol, rot @ c + off, rot @ inertia @ rot.T


def _shift(inertia_c: np.ndarray, mass: float, r: np.ndarray) -> np.ndarray:
    """Parallel-axis theorem: inertia about the origin from one about the
    CoG at r (same units: mass * length^2)."""
    return inertia_c + mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))


def _body(solids, measures):
    """Unit-density volume, centroid and inertia about the origin."""
    vol, first, about_o = 0.0, np.zeros(3), np.zeros((3, 3))
    for s in solids:
        v, c, ic = _placed(s, measures)
        vol += v
        first += v * c
        about_o += _shift(ic, v, c)
    centroid = first / vol if vol > 0 else np.zeros(3)
    return vol, centroid, about_o


def _r(a, nd):
    return (np.round(np.asarray(a, dtype=float), nd) + 0.0).tolist()  # no -0.0


def _inertia_report(i_kgm2: np.ndarray) -> dict:
    """Tensor components, 4 significant digits; entries below 1e-6 of the
    largest are integration noise (symmetric parts) and read 0."""
    keys = {"Ixx": (0, 0), "Iyy": (1, 1), "Izz": (2, 2),
            "Ixy": (0, 1), "Ixz": (0, 2), "Iyz": (1, 2)}
    floor = 1e-6 * float(np.abs(i_kgm2).max(initial=0.0))
    out = {}
    for k, (r, c) in keys.items():
        v = float(i_kgm2[r, c])
        out[k] = 0.0 if abs(v) <= floor else float(f"{v:.4g}")
    return out


def payload_limit(robot: str, offset_mm: float) -> tuple[float, bool]:
    """(allowed payload kg, on_chart) at a CoG distance from the flange."""
    key = kernel.norm(robot).replace(" ", "")
    if key not in PAYLOAD_CURVES:
        raise ValueError(f"No payload curve for robot {robot!r}; known: "
                         f"{sorted(PAYLOAD_CURVES)}")
    pts = PAYLOAD_CURVES[key]
    xs = [p[0] for p in pts]
    if offset_mm >= xs[-1]:
        return pts[-1][1], offset_mm <= xs[-1]
    i = bisect.bisect_right(xs, offset_mm)
    (x0, y0), (x1, y1) = pts[i - 1], pts[i]
    return y0 + (y1 - y0) * (offset_mm - x0) / (x1 - x0), True


def robot_check(robot: str, mass_kg: float, cog_mm) -> dict:
    offset = float(np.linalg.norm(cog_mm))
    allowed, on_chart = payload_limit(robot, offset)
    curve = PAYLOAD_CURVES[kernel.norm(robot).replace(" ", "")]
    rated = curve[0][1]
    out = {"robot": robot, "rated_payload_kg": rated,
           "cog_offset_mm": round(offset, 1),
           "allowed_at_offset_kg": round(allowed, 2),
           "utilisation": round(mass_kg / allowed, 3),
           "ok": bool(on_chart and mass_kg <= allowed)}
    warnings = []
    if not on_chart:
        warnings.append(f"CoG offset {offset:.0f} mm is beyond the {robot} payload "
                        f"curve ({curve[-1][0]} mm); "
                        "check with Universal Robots before use")
    if mass_kg > allowed:
        warnings.append(f"payload {mass_kg:.2f} kg exceeds the {allowed:.2f} kg "
                        f"allowed at {offset:.0f} mm CoG offset")
    elif mass_kg > 0.9 * allowed:
        warnings.append(f"payload {mass_kg:.2f} kg is within 10 % of the "
                        f"{allowed:.2f} kg limit at {offset:.0f} mm")
    out["warnings"] = warnings
    return out


def mass_properties(shape, densities: dict | None = None,
                    vendor_masses: dict | None = None,
                    exclude: list[str] | None = None,
                    robot: str | None = None) -> dict:
    """Per-part and total mass / CoG / inertia of a labelled assembly.

    densities: label substring -> material name or g/cm^3 ('*' = default).
    vendor_masses: label substring -> kg per matching part (wins over
        densities); density is scaled to hit it, CoG/inertia keep the shape.
    exclude: label substrings of parts that are not payload (the robot).
    """
    if exclude is None and robot:
        exclude = [robot]  # the robot's own CAD is not payload
    shape = kernel.unwrap_groups(shape)
    names = kernel.child_names(shape, "part")
    if names:
        groups = list(zip(names, shape.children))
    else:
        groups = [("model", shape)]
    for spec in (densities or {}).values():
        density(spec)  # fail fast on a bad material, before integrating
    excluded = [n for n, _ in groups
                if any(kernel.norm(p) and kernel.norm(p) in kernel.norm(n) for p in exclude or [])]
    todo = [(n, c) for n, c in groups if n not in excluded]
    unassigned = [n for n, _ in todo
                  if _match(n, vendor_masses) is None and _match(n, densities) is None]
    if unassigned:
        raise ValueError(
            f"No material for part(s) {unassigned}. Add them to densities "
            "(label substring -> material, '*' for a default), vendor_masses "
            f"or exclude. Parts: {[n for n, _ in groups]}")

    parts = []
    total_m, total_first, total_io = 0.0, np.zeros(3), np.zeros((3, 3))
    with kernel.Measures() as measures:
        for name, child in todo:
            solids = list(child.solids())
            vol, centroid, io_vol = _body(solids, measures)
            vendor = _match(name, vendor_masses)
            if vendor is not None:
                mass = float(vendor)
                rho_kg_mm3 = mass / vol if vol > 0 else 0.0
                source = f"vendor mass {mass:g} kg"
            else:
                spec = _match(name, densities)
                rho = density(spec)
                rho_kg_mm3 = rho * 1e-6
                mass = vol * rho_kg_mm3
                source = f"{spec} ({rho:g} g/cm3)" if not isinstance(spec, (int, float)) \
                    else f"{rho:g} g/cm3"
            io = io_vol * rho_kg_mm3 * 1e-6  # kg*mm^2 -> kg*m^2
            total_m += mass
            total_first += mass * centroid
            total_io += io
            parts.append({
                "part": name, "solids": len(solids), "material": source,
                "volume_cm3": round(vol / 1000, 2),
                "density_g_cm3": round(rho_kg_mm3 * 1e6, 3),
                "mass_kg": round(mass, 4),
                "cog_mm": _r(centroid, 2),
                "inertia_origin_kgm2": _inertia_report(io),
            })

    cog = total_first / total_m if total_m > 0 else np.zeros(3)
    i_cog = total_io - _shift(np.zeros((3, 3)), total_m, cog * 1e-3)
    out = {
        "units": {"mass": "kg", "cog": "mm", "inertia": "kg*m^2"},
        "frame": "file world frame (tool flange for end-effector models)",
        "parts": parts,
        "total": {
            "mass_kg": round(total_m, 3),
            "cog_mm": _r(cog, 1),
            "inertia_origin_kgm2": _inertia_report(total_io),
            "inertia_cog_kgm2": _inertia_report(i_cog),
        },
    }
    if excluded:
        out["excluded"] = excluded
    if robot:
        out["robot_check"] = robot_check(robot, total_m, cog)
    return out
