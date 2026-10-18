"""Beam FE analysis of welded tube frames (robot stands, carts, racks).

3D Euler-Bernoulli beams (12x12 local stiffness), EN 10219 hollow sections,
automatic joints with rigid offset links, unit-load superposition for load
case envelopes, von Mises stress per element and Guyan-condensed
eigenfrequencies with mode-shape recovery.

Units inside: mm, N, N*mm, kg.  The public spec takes forces in N and
moments in N*m (converted here).  Results are frame-only: floor, castor and
weld compliance are not modelled, joints are rigid.

Spec (all keys but `members` optional):
    members:  [{name, p0, p1, section}]  section = [w, t] (square tube),
              [b, h, t] (rectangular; b along local y = up x axis, i.e. the
              horizontal width of a horizontal member) or "rigid" (stiff
              massless link, e.g. a robot base plate or adapter stub).
    supports: [point | {point, fix}]  fix "xyz" (pinned, default) or "all".
    masses:   [{point, kg, name?}]
    loads:    [{point, F?: [N]*3, M?: [Nm]*3, case?, name?}] - loads with the
              same case name act together (default case "load").
    sweeps:   [{point, Fxy?, Fz?, Mxy?, Mz?, step_deg?, case?, name?}] - the
              UR "Dimensioning the stand" envelope: Fxy and Mxy rotated in
              step_deg steps independently, Fz and Mz with both signs.
    probes:   [{point, name}] extra points for nodal stiffness (load points
              are probes automatically).
A point that lies on a member (or a rigid member end) becomes a node there;
any other point is tied to the nearest member by a rigid link.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STEEL = dict(E=210e3, G=81e3, rho=7.85e-6, fy=235.0)    # N/mm2, N/mm2, kg/mm3, N/mm2
RIGID = dict(A=1e5, Iy=1e10, Iz=1e10, J=1e10, Wy=1e9, Wz=1e9, Wt=1e9, b=0, h=0, t=0)
DOF = ("x", "y", "z", "rx", "ry", "rz")


def section(spec) -> dict:
    """EN 10219 hollow section props from [w, t] or [b, h, t] (mm), or "rigid".
    Corner radii are ignored (slightly conservative)."""
    if isinstance(spec, str):
        if spec.lower() == "rigid":
            return dict(RIGID)
        raise ValueError(f"Unknown section {spec!r}: use [w, t], [b, h, t] or 'rigid'")
    spec = [float(v) for v in spec]
    if len(spec) == 2:
        b = h = spec[0]; t = spec[1]
    elif len(spec) == 3:
        b, h, t = spec                      # b along local y (= up x d), h along local z
    else:
        raise ValueError(f"Section {spec!r}: use [w, t] or [b, h, t]")
    if not 0 < 2 * t < min(b, h):
        raise ValueError(f"Section {spec!r}: wall t must be below half the width")
    A = b * h - (b - 2 * t) * (h - 2 * t)
    Iy = (b * h ** 3 - (b - 2 * t) * (h - 2 * t) ** 3) / 12      # bending about local y (uses h)
    Iz = (h * b ** 3 - (h - 2 * t) * (b - 2 * t) ** 3) / 12
    Am = (b - t) * (h - t); J = 4 * Am ** 2 * t / (2 * ((b - t) + (h - t)))   # Bredt
    return dict(A=A, Iy=Iy, Iz=Iz, J=J, Wy=Iy / (h / 2), Wz=Iz / (b / 2), Wt=2 * Am * t, b=b, h=h, t=t)


def frame_axes(d):
    """Rows: local x (= member axis d), y, z.  y = up x d with up = Z, or X
    for (near-)vertical members."""
    up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0, 0])
    y = np.cross(up, d); y /= np.linalg.norm(y)
    z = np.cross(d, y)
    return np.vstack([d, y, z])


def k_local(L, s, E=STEEL["E"], G=STEEL["G"]):
    A, Iy, Iz, J = s["A"], s["Iy"], s["Iz"], s["J"]
    k = np.zeros((12, 12))
    ea, gj = E * A / L, G * J / L
    k[0, 0] = k[6, 6] = ea; k[0, 6] = k[6, 0] = -ea
    k[3, 3] = k[9, 9] = gj; k[3, 9] = k[9, 3] = -gj
    for (v, r, I, sg) in ((1, 5, Iz, 1), (2, 4, Iy, -1)):
        a = 12 * E * I / L ** 3; b = 6 * E * I / L ** 2; c = 4 * E * I / L; d = 2 * E * I / L
        idx = [v, r, v + 6, r + 6]
        m = np.array([[a, sg * b, -a, sg * b], [sg * b, c, -sg * b, d],
                      [-a, -sg * b, a, -sg * b], [sg * b, d, -sg * b, c]])
        for i in range(4):
            for j in range(4):
                k[idx[i], idx[j]] = m[i, j]
    return k


def closest_on_segment(p, a, b):
    d = b - a; t = float(np.clip((p - a) @ d / (d @ d), 0, 1))
    return t, a + t * d


@dataclass
class Model:
    X: list = field(default_factory=list)      # node coordinates (mm)
    el: list = field(default_factory=list)     # (a, b, section, name, steel)
    fixed: list = field(default_factory=list)  # constrained global DOFs
    points: dict = field(default_factory=dict)  # label -> node
    mat: dict = field(default_factory=lambda: dict(STEEL))

    def node(self, p, tol=0.5):
        p = np.asarray(p, float)
        for i, q in enumerate(self.X):
            if np.linalg.norm(q - p) < tol:
                return i
        self.X.append(p); return len(self.X) - 1

    def beam(self, a, b, s, name, steel=True):
        self.el.append((a, b, s, name, steel))


def _pt(p, what):
    p = np.asarray(p, float)
    if p.shape != (3,):
        raise ValueError(f"{what}: point must be [x, y, z] in mm, got {p.tolist()}")
    return p


def build(members, supports=(), key_points=(), max_element_mm=120.0, joint_gap_mm=5.0,
          mat=None) -> Model:
    """Mesh the frame.  key_points: [(label, point)] that must become nodes
    (masses, loads, probes); supports as in the module spec."""
    steel, rigid = {}, []
    for m in members:
        name = m["name"]
        a, b = _pt(m["p0"], name), _pt(m["p1"], name)
        if np.linalg.norm(b - a) < 1e-6:
            raise ValueError(f"Member {name!r} has zero length")
        s = section(m["section"])
        if s["b"] == 0:
            rigid.append((a, b, s, name))
        else:
            if name in steel:
                raise ValueError(f"Duplicate member name {name!r}")
            steel[name] = (a, b, s)
    if not steel:
        raise ValueError("No steel members (only rigid links)")
    cuts = {k: [0.0, 1.0] for k in steel}

    def nearest(p, skip=None):
        best = None
        for k, (b0, b1, sb) in steel.items():
            if k == skip:
                continue
            t, q = closest_on_segment(p, b0, b1)
            dist = float(np.linalg.norm(q - p))
            if best is None or dist < best[0]:
                best = (dist, k, t, q)
        return best

    # automatic joints: each member end to the nearest other member within reach
    links = []
    for ka, (a0, a1, sa) in steel.items():
        for end in (a0, a1):
            best = None
            for kb, (b0, b1, sb) in steel.items():
                if kb == ka:
                    continue
                t, q = closest_on_segment(end, b0, b1)
                dist = np.linalg.norm(q - end)
                reach = (max(sa["b"], sa["h"]) + max(sb["b"], sb["h"])) / 2 + joint_gap_mm
                if dist <= reach and (best is None or dist < best[0]):
                    best = (dist, kb, t, q)
            if best:
                links.append((end, best[3]))
                cuts[best[1]].append(best[2])

    # rigid member ends and attachment points: on a member -> cut it there,
    # else (not a rigid end either) -> rigid link from the nearest member
    rigid_ends = [p for a, b, _, _ in rigid for p in (a, b)]
    sup = []
    for s_ in supports:
        p, fix = (s_, "xyz") if not isinstance(s_, dict) else (s_["point"], s_.get("fix", "xyz"))
        if fix not in ("xyz", "all"):
            raise ValueError(f"Support fix {fix!r}: use 'xyz' (pinned) or 'all' (clamped)")
        sup.append((_pt(p, "support"), fix))
    attach = [("", p) for p in rigid_ends] + [("", p) for p, _ in sup] + \
             [(lbl, _pt(p, lbl)) for lbl, p in key_points]
    offsets = []
    for lbl, p in attach:
        dist, k, t, q = nearest(p)
        if dist <= 0.5:                    # cut every member through the point
            for kb, (b0, b1, _) in steel.items():
                tb, qb = closest_on_segment(p, b0, b1)
                if np.linalg.norm(qb - p) <= 0.5:
                    cuts[kb].append(tb)
        elif not any(np.linalg.norm(p - r) < 0.5 for r in rigid_ends):
            cuts[k].append(t); offsets.append((q, p))

    M = Model(mat={**STEEL, **(mat or {})})
    for k, (a0, a1, s) in steel.items():
        ts = sorted(set(round(t, 6) for t in cuts[k]))
        pts = []
        for t0, t1 in zip(ts, ts[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(a1 - a0) * (t1 - t0) / max_element_mm)))
            pts += [t0 + (t1 - t0) * i / n for i in range(n)]
        pts.append(ts[-1])
        ids = [M.node(a0 + t * (a1 - a0)) for t in pts]
        for i, j in zip(ids, ids[1:]):
            M.beam(i, j, s, k)
    for end, q in links:
        i, j = M.node(end), M.node(q)
        if i != j:
            M.beam(i, j, RIGID, "link", steel=False)
    for a, b, s, name in rigid:
        M.beam(M.node(a), M.node(b), s, name, steel=False)
    for q, p in offsets:
        M.beam(M.node(q), M.node(p), RIGID, "link", steel=False)
    for p, fix in sup:
        n = M.node(p)
        M.fixed += [6 * n + i for i in range(3 if fix == "xyz" else 6)]
    M.fixed = sorted(set(M.fixed))
    for lbl, p in key_points:
        M.points[lbl] = M.node(p)
    return M


def assemble(M):
    E, G, rho = M.mat["E"], M.mat["G"], M.mat["rho"]
    n = len(M.X); K = np.zeros((6 * n, 6 * n)); mass = np.zeros(n)
    for a, b, s, name, steel in M.el:
        d = M.X[b] - M.X[a]; L = np.linalg.norm(d); R = frame_axes(d / L)
        T = np.kron(np.eye(4), R); k = T.T @ k_local(L, s, E, G) @ T
        idx = np.r_[6 * a:6 * a + 6, 6 * b:6 * b + 6]
        K[np.ix_(idx, idx)] += k
        if steel:
            mass[[a, b]] += s["A"] * L * rho / 2
    return K, mass


def solve(M, K, loads):
    """Displacements for load columns (6n x m); constrained DOFs stay 0."""
    n = len(M.X)
    free = np.setdiff1d(np.arange(6 * n), M.fixed)
    U = np.zeros((6 * n, loads.shape[1]))
    try:
        U[free] = np.linalg.solve(K[np.ix_(free, free)], loads[free])
    except np.linalg.LinAlgError:
        raise ValueError("Stiffness matrix is singular: the frame is a mechanism or "
                         "under-supported (need >= 3 non-collinear pinned supports)") from None
    if not np.isfinite(U).all() or np.abs(U).max() > 1e6:
        raise ValueError("Frame is (nearly) a mechanism: check supports and joints")
    return U


def element_stress(M, U):
    """Max von Mises (nominal, box section) per steel element for each load
    column: [(member name, array over columns)]."""
    E, G = M.mat["E"], M.mat["G"]
    out = []
    for a, b, s, name, steel in M.el:
        if not steel:
            continue
        d = M.X[b] - M.X[a]; L = np.linalg.norm(d); R = frame_axes(d / L)
        T = np.kron(np.eye(4), R)
        f = k_local(L, s, E, G) @ T @ np.vstack([U[6 * a:6 * a + 6], U[6 * b:6 * b + 6]])
        best = np.zeros(U.shape[1])
        for o in (0, 6):
            N, Vy, Vz, Tq, My, Mz = (f[o + i] for i in range(6))
            sig = np.abs(N) / s["A"] + np.abs(My) / s["Wy"] + np.abs(Mz) / s["Wz"]
            tau = np.abs(Tq) / s["Wt"] + np.maximum(np.abs(Vy), np.abs(Vz)) / (s["A"] / 2)
            best = np.maximum(best, np.sqrt(sig ** 2 + 3 * tau ** 2))
        out.append((name, best))
    return out


def rigid_body_check(M, K):
    """Relative residual of K under rigid translations and rotations (must be ~0)."""
    n = len(M.X); X = np.array(M.X); worst = 0.0
    for ax in range(3):
        r = np.zeros(6 * n); r[ax::6] = 1.0
        worst = max(worst, np.abs(K @ r).max())
        w = np.zeros(3); w[ax] = 1e-3                   # small rotation about the origin
        r = np.zeros(6 * n)
        u = np.cross(w, X)
        for i in range(3):
            r[i::6] = u[:, i]
        r[3 + ax::6] = 1e-3
        worst = max(worst, np.abs(K @ r).max() / 1e-3 / max(1.0, np.abs(X).max()))
    return worst / np.abs(K).max()


def modes(M, K, mass, n_modes=3):
    """Eigenfrequencies (Hz) and normalised translational mode shapes (n x 3),
    Guyan-condensed onto the translations of nodes with mass."""
    n = len(M.X); fixed = set(M.fixed)
    t = [i for i in range(6 * n) if i % 6 < 3 and i not in fixed and mass[i // 6] > 1e-6]
    ts = set(t)
    r = [i for i in range(6 * n) if i not in fixed and i not in ts]   # condensed out
    Krt = K[np.ix_(r, t)]
    Kc = K[np.ix_(t, t)] - Krt.T @ np.linalg.solve(K[np.ix_(r, r)], Krt)
    D = 1 / np.sqrt(np.array([mass[i // 6] for i in t]))         # kg; K in N/mm -> *1000 for N/m
    w2, V = np.linalg.eigh(D[:, None] * Kc * D[None, :] * 1000.0)
    keep = np.flatnonzero(w2 > 1e-6)[:n_modes]
    out = []
    for k in keep:
        full = np.zeros(6 * n)
        full[t] = D * V[:, k]
        full[r] = -np.linalg.solve(K[np.ix_(r, r)], K[np.ix_(r, t)] @ full[t])   # Guyan recovery
        u = full.reshape(-1, 6)[:, :3]
        out.append((float(np.sqrt(w2[k]) / 2 / np.pi), u / np.abs(u).max()))
    return out


def sweep_combos(Fxy=0.0, Fz=0.0, Mxy=0.0, Mz=0.0, step_deg=30):
    """UR stand envelope: [Fx Fy Fz Mx My Mz] rows (N, N*mm) and (phi, psi,
    sign Fz, sign Mz) tags.  Zero components are not doubled by sign."""
    rows, tags = [], []
    ang = range(0, 360, int(step_deg))
    for phi, psi, sz, sm in itertools.product(ang if Fxy else [0], ang if Mxy else [0],
                                              (-1, 1) if Fz else [1], (-1, 1) if Mz else [1]):
        f, p = np.radians(phi), np.radians(psi)
        rows.append([Fxy * np.cos(f), Fxy * np.sin(f), sz * Fz,
                     Mxy * np.cos(p), Mxy * np.sin(p), sm * Mz])
        tags.append((phi, psi, sz, sm))
    return np.array(rows), tags


@dataclass
class Case:
    name: str
    C: np.ndarray            # (6 * n_load_points) x n_combos, N / N*mm
    tags: list               # one per combo (None for a plain case)
    U: np.ndarray = None     # 6n x n_combos
    stress: list = None      # [(member, array over combos)]

    def max_stress(self):
        name, s = max(self.stress, key=lambda e: e[1].max())
        k = int(np.argmax(s))
        return float(s[k]), name, k


@dataclass
class Result:
    M: Model
    K: np.ndarray
    mass: np.ndarray
    unit: np.ndarray         # 6n x 6*len(load_labels): unit loads at each load/probe node
    labels: list             # load/probe labels, in unit column order
    cases: dict
    modes: list
    rigid_residual: float
    point_masses: dict

    def unit_at(self, label):
        """Displacements (6n x 6) for unit Fx Fy Fz Mx My Mz (N, N*mm) at `label`."""
        i = self.labels.index(label)
        return self.unit[:, 6 * i:6 * i + 6]

    def stiffness(self, label):
        """Direct nodal stiffness per DOF: N/mm for x y z, kNm/rad for rotations."""
        n = self.M.points[label]
        c = np.diag(self.unit_at(label)[6 * n:6 * n + 6])
        k = 1 / np.maximum(np.abs(c), 1e-30)
        return {d: (round(float(k[i]), 1) if i < 3 else round(float(k[i]) / 1e6, 1))
                for i, d in enumerate(DOF)}


def _combo_matrix(labels, rows, label):
    i = labels.index(label)
    C = np.zeros((6 * len(labels), len(rows)))
    C[6 * i:6 * i + 6] = np.asarray(rows, float).T
    return C


def analyse(spec: dict, n_modes=3) -> Result:
    """Run the full analysis of a frame spec (see module docstring)."""
    def label(d, prefix, i):
        return str(d.get("name") or f"{prefix}{i + 1}")

    masses = [(label(m, "mass", i), _pt(m["point"], "mass"), float(m["kg"]))
              for i, m in enumerate(spec.get("masses", ()))]
    loads = [dict(l, _label=label(l, "load", i)) for i, l in enumerate(spec.get("loads", ()))]
    sweeps = [dict(s, _label=label(s, "sweep", i)) for i, s in enumerate(spec.get("sweeps", ()))]
    probes = [(str(p["name"]), _pt(p["point"], p["name"])) for p in spec.get("probes", ())]
    if not spec.get("supports"):
        raise ValueError("No supports given")
    labels, pts = [], {}
    for lbl, p in [(l["_label"], l["point"]) for l in loads + sweeps] + probes:
        p = _pt(p, lbl)
        if lbl in pts and np.linalg.norm(pts[lbl] - p) > 0.5:
            raise ValueError(f"Point name {lbl!r} used for two different points")
        if lbl not in pts:
            labels.append(lbl); pts[lbl] = p
    key = [(lbl, p) for lbl, p, _ in masses] + [(lbl, pts[lbl]) for lbl in labels]
    M = build(spec["members"], spec["supports"], key, spec.get("max_element_mm", 120.0),
              spec.get("joint_gap_mm", 5.0), spec.get("material"))
    K, mass = assemble(M)
    resid = rigid_body_check(M, K)
    if resid > 1e-9:
        raise RuntimeError(f"Rigid-body check failed (residual {resid:.1e}): stiffness is wrong")
    m_all = mass.copy()
    for lbl, _, kg in masses:
        m_all[M.points[lbl]] += kg

    n = len(M.X)
    unit = np.zeros((6 * n, 6 * len(labels)))
    for i, lbl in enumerate(labels):
        for j in range(6):
            unit[6 * M.points[lbl] + j, 6 * i + j] = 1.0
    U1 = solve(M, K, unit) if labels else unit

    cases = {}
    for l in loads:
        F = np.asarray(l.get("F", (0, 0, 0)), float); Mo = np.asarray(l.get("M", (0, 0, 0)), float) * 1e3
        C = _combo_matrix(labels, [np.r_[F, Mo]], l["_label"])
        cn = str(l.get("case", "load"))
        if cn in cases:
            cases[cn].C = cases[cn].C + C
        else:
            cases[cn] = Case(cn, C, [None])
    for s in sweeps:
        cn = str(s.get("case") or s["_label"])
        if cn in cases:
            raise ValueError(f"Sweep case {cn!r} already exists: give each sweep its own case")
        rows, tags = sweep_combos(s.get("Fxy", 0.0), s.get("Fz", 0.0), s.get("Mxy", 0.0) * 1e3,
                                  s.get("Mz", 0.0) * 1e3, s.get("step_deg", 30))
        cases[cn] = Case(cn, _combo_matrix(labels, rows, s["_label"]), tags)
    for c in cases.values():
        c.U = U1 @ c.C
        c.stress = element_stress(M, c.U)
    return Result(M, K, m_all, U1, labels, cases, modes(M, K, m_all, n_modes) if n_modes else [],
                  resid, {lbl: kg for lbl, _, kg in masses})


def summary(R: Result) -> dict:
    """JSON-ready results: max stress, critical member, stiffness, frequencies."""
    M, fy = R.M, R.M.mat["fy"]
    out = {"model": {"nodes": len(M.X), "elements": len(M.el),
                     "steel_mass_kg": round(float(R.mass.sum() - sum(R.point_masses.values())), 1),
                     "point_masses_kg": R.point_masses,
                     "rigid_body_residual": float(f"{R.rigid_residual:.1e}")},
           "cases": {}}
    worst = None
    for c in R.cases.values():
        smax, member, k = c.max_stress()
        per_member = {}
        for name, s in c.stress:
            per_member[name] = max(per_member.get(name, 0.0), float(s.max()))
        disp = {}
        for lbl in R.labels:
            u = c.U[6 * M.points[lbl]:6 * M.points[lbl] + 6]
            disp[lbl] = {"max_translation_mm": round(float(np.linalg.norm(u[:3], axis=0).max()), 3),
                         "max_rotation_mrad": round(float(np.linalg.norm(u[3:], axis=0).max()) * 1e3, 3)}
        tag = c.tags[k]
        out["cases"][c.name] = {
            "combinations": len(c.tags),
            "max_von_mises_MPa": round(smax, 1), "utilisation_fy": round(smax / fy, 2),
            "critical_member": member,
            "critical_combination": (None if tag is None else
                                     dict(zip(("Fxy_angle_deg", "Mxy_angle_deg", "Fz_sign", "Mz_sign"), tag))),
            "per_member_max_MPa": {k_: round(v, 1) for k_, v in
                                   sorted(per_member.items(), key=lambda kv: -kv[1])},
            "displacements": disp}
        if worst is None or smax > worst[0]:
            worst = (smax, c.name, member)
    if worst:
        out["max_stress"] = {"MPa": round(worst[0], 1), "utilisation_fy": round(worst[0] / fy, 2),
                             "case": worst[1], "member": worst[2]}
    out["stiffness"] = {lbl: R.stiffness(lbl) for lbl in R.labels}
    out["stiffness_units"] = "x y z: N/mm, rx ry rz: kNm/rad (direct, at the node)"
    out["eigenfrequencies_Hz"] = [round(f, 1) for f, _ in R.modes]
    return out


# ---------------------------------------------------------------------------
# plots

def _setup(ax, X, title, view=(22, -58), aspect=None):
    ax.set_title(title, fontsize=9)
    ax.set_box_aspect(tuple(aspect) if aspect else tuple(np.maximum(np.ptp(X, axis=0), 1.0)))
    ax.view_init(elev=view[0], azim=view[1]); ax.set_axis_off()


def plot_stress(R, case, path, title=None, view=(22, -58), aspect=None):
    """Members coloured by their envelope von Mises stress in `case`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    M = R.M; X = np.array(M.X); c = R.cases[case]
    steel = [(a, b) for a, b, _, _, st in M.el if st]
    smax = {ab: float(sv.max()) for ab, (_, sv) in zip(steel, c.stress)}
    fig = plt.figure(figsize=(7.2, 6.0)); ax = fig.add_subplot(111, projection="3d")
    norm = colors.Normalize(0, max(smax.values()) or 1.0); cmap = cm.turbo
    for (a, b), v in smax.items():
        ax.plot(*X[[a, b]].T, color=cmap(norm(v)), lw=3.2)
    for a, b, _, _, st in M.el:
        if not st:
            ax.plot(*X[[a, b]].T, color="0.35", lw=1.0)       # rigid joint offsets / links
    for i in sorted({d // 6 for d in M.fixed}):
        ax.scatter(*X[i], marker="^", s=40, color="k")
    for lbl in R.labels:
        ax.scatter(*X[M.points[lbl]], s=60, color="r")
    _setup(ax, X, title or f"von Mises stress, envelope of case '{case}'", view, aspect)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); fig.colorbar(sm, ax=ax, shrink=0.6, label="σv [MPa]")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)
    return str(path)


def plot_deformation(R, case, path, combo=None, scale=None, title=None, view=(22, -58),
                     aspect=None):
    """Exaggerated deformed shape for one combination (default: the one with
    the largest nodal translation)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = R.M; X = np.array(M.X); c = R.cases[case]
    Ut = c.U.reshape(len(M.X), 6, -1)[:, :3]                  # node, xyz, combo
    if combo is None:
        combo = int(np.argmax(np.linalg.norm(Ut, axis=1).max(axis=0)))
    U = Ut[:, :, combo]
    if scale is None:
        umax = float(np.linalg.norm(U, axis=1).max())
        scale = float(f"{0.05 * np.ptp(X, axis=0).max() / max(umax, 1e-12):.1g}")
    fig = plt.figure(figsize=(7.2, 6.0)); ax = fig.add_subplot(111, projection="3d")
    for a, b, _, _, _ in M.el:
        ax.plot(*X[[a, b]].T, color="0.75", lw=1.5)
        ax.plot(*(X[[a, b]] + scale * U[[a, b]]).T, color="tab:red", lw=2.2)
    for lbl in R.labels:
        ax.scatter(*X[M.points[lbl]], s=50, color="k")
    _setup(ax, X, title or f"Deformation x{scale:g}, case '{case}' combination {combo}", view, aspect)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)
    return str(path)


def plot_modes(R, path, scale=None, view=(22, -58), aspect=None,
               title="Mode {k}: {f:.1f} Hz"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = R.M; X = np.array(M.X)
    if scale is None:
        scale = 0.09 * np.ptp(X, axis=0).max()
    nm = max(1, len(R.modes))
    fig = plt.figure(figsize=(3.5 * nm, 4.2))
    for k, (f, u) in enumerate(R.modes):
        ax = fig.add_subplot(1, nm, k + 1, projection="3d")
        for a, b, _, _, _ in M.el:
            ax.plot(*X[[a, b]].T, color="0.8", lw=1.0)
            ax.plot(*(X[[a, b]] + scale * u[[a, b]]).T, color="tab:blue", lw=1.8)
        _setup(ax, X, title.format(k=k + 1, f=f), view, aspect)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)
    return str(path)


def plot_all(R, outdir, prefix="frame") -> dict:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    out = {}
    if R.cases:
        worst = max(R.cases.values(), key=lambda c: c.max_stress()[0]).name
        out["stress"] = plot_stress(R, worst, outdir / f"{prefix}_stress.png")
        out["deformation"] = plot_deformation(R, worst, outdir / f"{prefix}_deformation.png")
    if R.modes:
        out["modes"] = plot_modes(R, outdir / f"{prefix}_modes.png")
    return out
