"""2-D manufacturing drawings from B-rep solids (A3, DIN-style title block).

Hidden-line views come from OpenCascade's HLR, every number is measured from
the solid (never typed), and each drawn dimension is recorded in
``Sheet.dims`` so callers can cross-check the drawing against the model.

Views follow ISO first-angle projection (Projektionsmethode 1, Europe):
top view below the front view, view from the left to the right of it.

Used by the ``create_drawing`` MCP tool (one part -> one PDF) and by project
scripts that build multi-sheet drawing sets (parts lists, weld assemblies).
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

from build123d import GeomType
from build123d.topology import Edge
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCP.HLRAlgo import HLRAlgo_Projector
from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_EDGE
from OCP.BRepLib import BRepLib

A3 = (420.0, 297.0)
STD_SCALES = [1, 2, 5, 10, 20, 50]          # 1:n
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})

# view name -> (direction towards the viewer, sheet x axis, sheet y axis)
VIEWS = {
    "vorne": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    "hinten": ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    "oben": ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    "unten": ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
    "links": ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    "rechts": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
}
VIEW_TITLE = {"vorne": "Vorderansicht", "hinten": "Rückansicht", "oben": "Draufsicht",
              "unten": "Untersicht", "links": "Seitenansicht von links",
              "rechts": "Seitenansicht von rechts"}


def mm(v) -> str:
    """German number format, 0.1 mm."""
    v = round(float(v), 1)
    return f"{v:g}".replace(".", ",")


def basis(view: str):
    d, x, y = VIEWS[view]
    return np.array(d, float), np.array(x, float), np.array(y, float)


def proj(p, view: str) -> np.ndarray:
    _, x, y = basis(view)
    p = np.asarray(p, float)
    return np.array([p @ x, p @ y])


def hlr(shape, view: str) -> dict:
    """Visible / hidden edges of `shape` seen from `view`, as 2-D polylines in mm."""
    d, x, _ = basis(view)
    algo = HLRBRep_Algo(); algo.Add(shape.wrapped)
    algo.Projector(HLRAlgo_Projector(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(*d), gp_Dir(*x))))
    algo.Update(); algo.Hide()
    h = HLRBRep_HLRToShape(algo)
    out = {"vis": [], "hid": []}
    for key, comps in (("vis", (h.VCompound(), h.OutLineVCompound())),
                       ("hid", (h.HCompound(), h.OutLineHCompound()))):
        for c in comps:
            if c.IsNull():
                continue
            exp = TopExp_Explorer(c, TopAbs_EDGE)
            while exp.More():
                e = exp.Current(); BRepLib.BuildCurves3d_s(e, 1e-4)
                E = Edge(e)
                n = 2 if E.geom_type == GeomType.LINE else 32
                pts = [E.position_at(t) for t in np.linspace(0, 1, n)]
                out[key].append(np.array([[p.X, p.Y] for p in pts]))
                exp.Next()
    return out


def extent(shape, view: str) -> tuple[np.ndarray, np.ndarray]:
    """2-D bounding box of the projected shape (from its 3-D bbox corners)."""
    bb = shape.bounding_box()
    cs = np.array([[x, y, z] for x in (bb.min.X, bb.max.X) for y in (bb.min.Y, bb.max.Y)
                   for z in (bb.min.Z, bb.max.Z)])
    P = np.array([proj(c, view) for c in cs])
    return P.min(0), P.max(0)


def hole_center(face):
    """Axis point of a cylindrical bore = mean centre of its circular edges.
    (Face.center() of a half-cylinder bore is off the axis by the radius.)"""
    cs = [e.arc_center for e in face.edges() if e.geom_type == GeomType.CIRCLE]
    if not cs:
        raise ValueError("bore face without circular edge")
    return sum(cs[1:], cs[0]) * (1.0 / len(cs))


def holes(shape, max_dia: float = 40.0) -> list[dict]:
    """Cylindrical bores of a solid: centre, axis, Ø (deduplicated half-faces)."""
    found = {}
    for f in shape.faces():
        if f.geom_type != GeomType.CYLINDER or not f.radius or 2 * f.radius > max_dia:
            continue
        try:
            c = hole_center(f)
        except ValueError:
            continue
        key = (round(c.X, 2), round(c.Y, 2), round(c.Z, 2), round(2 * f.radius, 2))
        found[key] = {"x": key[0], "y": key[1], "z": key[2], "d": key[3]}
    return sorted(found.values(), key=lambda h: (h["d"], h["x"], h["y"], h["z"]))


def pick_scale(sizes_mm, area_mm) -> int:
    """Largest standard reduction 1:n so that `sizes` fit in `area`."""
    for n in STD_SCALES:
        if all(s / n <= a for s, a in zip(sizes_mm, area_mm)):
            return n
    return STD_SCALES[-1]


@dataclass
class TitleInfo:
    title: str
    number: str
    project: str = ""
    material: str = "siehe Stückliste"
    scale: str = ""
    tolerances: str = "Schweißteile ISO 13920-BF · Bearbeitung ISO 2768-mK"
    author: str = "CAD-generiert"
    date: str = field(default_factory=lambda: datetime.date.today().strftime("%d.%m.%Y"))


class Sheet:
    """One A3 landscape sheet with frame and title block. Coordinates in mm."""

    def __init__(self, info: TitleInfo, sheet_no: int = 1):
        self.info, self.no = info, sheet_no
        self.dims: list[dict] = []
        self.fig = plt.figure(figsize=(A3[0] / 25.4, A3[1] / 25.4))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        ax = self.ax
        ax.set_xlim(0, A3[0]); ax.set_ylim(0, A3[1]); ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(Rectangle((20, 10), 390, 277, fill=False, lw=0.7))
        self._title_block()

    # drawing area left free of the title block (x0, y0, x1, y1)
    AREA = (22, 52, 408, 285)

    def _title_block(self):
        i, ax = self.info, self.ax
        x0, y0, w, h = 230, 10, 180, 40
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, lw=0.7))
        rows = [("Projekt", i.project), ("Benennung", i.title),
                ("Zeichnungs-Nr.", f"{i.number}  ·  Blatt {self.no}"), ("Werkstoff", i.material),
                ("Maßstab", f"{i.scale}  (Maße in mm, Projektionsmethode 1)"),
                ("Allg.-Toleranzen", i.tolerances),
                ("Erstellt / Datum", f"{i.author} · {i.date}"),
                ("Geprüft / Freigabe", "________________________   (vor Fertigung freigeben)")]
        for k, (lab, val) in enumerate(rows):
            y = y0 + h - (k + 1) * h / len(rows)
            ax.plot([x0, x0 + w], [y, y], lw=0.3, color="k")
            ax.text(x0 + 2, y + 1.3, lab, fontsize=5.5, color="0.35")
            ax.text(x0 + 32, y + 1.3, val, fontsize=7.5 if lab == "Benennung" else 6.5,
                    weight="bold" if lab in ("Benennung", "Zeichnungs-Nr.") else "normal")
        ax.plot([x0 + 30, x0 + 30], [y0, y0 + h], lw=0.3, color="k")

    def text(self, x, y, s, size=6.5, **kw):
        self.ax.text(x, y, s, fontsize=size, **kw)

    def note(self, x, y, lines, size=6.5):
        for k, l in enumerate(lines):
            self.ax.text(x, y - k * size * 0.55, l, fontsize=size)
        return y - len(lines) * size * 0.55

    def register(self, label, value, text=None):
        self.dims.append({"blatt": self.no, "bezeichnung": label, "wert": value, "text": text})

    def save(self, pdf):
        pdf.savefig(self.fig); plt.close(self.fig)


class View:
    """Projected view on a sheet. sheet_xy = origin + scale * view_xy, where the
    3-D point `ref3d` lands on `origin`."""

    def __init__(self, sheet: Sheet, shape, view: str, scale: float, origin,
                 ref3d=(0, 0, 0), hidden=True, lw=0.45, title=None):
        self.s, self.view, self.k = sheet, view, scale
        self.o = np.asarray(origin, float) - scale * proj(ref3d, view)
        e = hlr(shape, view)
        for pl in e["vis"]:
            P = self.o + scale * pl; sheet.ax.plot(P[:, 0], P[:, 1], color="k", lw=lw)
        if hidden:
            for pl in e["hid"]:
                P = self.o + scale * pl
                sheet.ax.plot(P[:, 0], P[:, 1], color="0.45", lw=0.25, ls=(0, (3, 2)))
        lo, hi = extent(shape, view)
        self.lo, self.hi = self.o + scale * lo, self.o + scale * hi
        if title:
            sheet.ax.text((self.lo[0] + self.hi[0]) / 2, self.lo[1] - 9, title,
                          ha="center", fontsize=7.5, weight="bold")

    def pt(self, p3):
        return self.o + self.k * proj(p3, self.view)

    def centre_lines(self, p3, r):
        c = self.pt(p3); r = r * self.k + 2
        for a, b in (((c[0] - r, c[1]), (c[0] + r, c[1])), ((c[0], c[1] - r), (c[0], c[1] + r))):
            self.s.ax.plot([a[0], b[0]], [a[1], b[1]], lw=0.2, color="0.3", ls="-.")


def dim(sheet: Sheet, a, b, off, text, horiz=None, size=6.5, register=None):
    """Linear dimension between sheet points a and b; `off` shifts the
    dimension line (sheet mm, sign = side)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if horiz is None:
        horiz = abs(b[0] - a[0]) >= abs(b[1] - a[1])
    ax = sheet.ax
    sg = np.sign(off) or 1
    if horiz:
        y = (max(a[1], b[1]) if off > 0 else min(a[1], b[1])) + off
        pa, pb = np.array([a[0], y]), np.array([b[0], y])
        for p, q in ((a, pa), (b, pb)):
            ax.plot([p[0], q[0]], [p[1] + sg * 1, q[1] + sg * 1.5], lw=0.25, color="k")
    else:
        x = (max(a[0], b[0]) if off > 0 else min(a[0], b[0])) + off
        pa, pb = np.array([x, a[1]]), np.array([x, b[1]])
        for p, q in ((a, pa), (b, pb)):
            ax.plot([p[0] + sg * 1, q[0] + sg * 1.5], [p[1], q[1]], lw=0.25, color="k")
    ax.add_patch(FancyArrowPatch(pa, pb, arrowstyle="<|-|>", mutation_scale=4, lw=0.3,
                                 color="k", shrinkA=0, shrinkB=0))
    m = (pa + pb) / 2
    ax.text(m[0] - (0.9 if not horiz else 0), m[1] + (0.7 if horiz else 0), text, fontsize=size,
            ha="center" if horiz else "right", va="bottom" if horiz else "center",
            rotation=0 if horiz else 90, rotation_mode="anchor",
            bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
    if register:
        sheet.register(register[0], register[1], text)


def balloon(sheet: Sheet, p, num, tip=None):
    ax = sheet.ax
    ax.add_patch(Circle(p, 3.2, fc="white", ec="k", lw=0.4, zorder=5))
    ax.text(p[0], p[1], str(num), ha="center", va="center", fontsize=6.5, weight="bold", zorder=6)
    if tip is not None:
        d = np.asarray(tip, float) - np.asarray(p, float)
        d /= max(np.linalg.norm(d), 1e-9)
        q = np.asarray(p) + d * 3.2
        ax.plot([q[0], tip[0]], [q[1], tip[1]], lw=0.3, color="k")
        ax.plot(tip[0], tip[1], "k.", ms=1.5)


def table(sheet: Sheet, x, y, cols, rows, widths, size=5.8, rh=4.2) -> float:
    """Simple ruled table; returns the y of its bottom edge."""
    ax = sheet.ax
    W = sum(widths)
    for i, row in enumerate([cols] + rows):
        yy = y - i * rh
        if i == 0:
            ax.add_patch(Rectangle((x, yy - rh), W, rh, fc="0.88", ec="none"))
        xx = x
        for c, w in zip(row, widths):
            ax.text(xx + 0.8, yy - rh + 1.2, str(c), fontsize=size,
                    weight="bold" if i == 0 else "normal")
            xx += w
        ax.plot([x, x + W], [yy - rh, yy - rh], lw=0.2, color="0.4")
    xx = x
    for w in [0] + list(widths):
        xx += w
        ax.plot([xx, xx], [y, y - (len(rows) + 1) * rh], lw=0.2, color="0.4")
    ax.plot([x, x + W], [y, y], lw=0.2, color="0.4")
    return y - (len(rows) + 1) * rh


def part_sheet(shape, info: TitleInfo, sheet_no=1, views=("vorne", "oben", "links"),
               scale: int | None = None, notes=()) -> Sheet:
    """Standard single-part drawing: three views in first-angle arrangement,
    overall dimensions on each view, hole table with coordinates from the
    part origin. Returns the Sheet (call .save(pdf))."""
    bb = shape.bounding_box()
    sx, sy, sz = bb.max.X - bb.min.X, bb.max.Y - bb.min.Y, bb.max.Z - bb.min.Z
    x0, y0, x1, y1 = Sheet.AREA
    n = scale or pick_scale([sx + sy, sz + sy], [(x1 - x0) * 0.62, (y1 - y0) * 0.78])
    k = 1.0 / n
    info.scale = info.scale or f"1:{n}"
    sh = Sheet(info, sheet_no)
    gap = 22
    # front view top-left, top view below, left view to the right (method 1)
    fx, fz = x0 + 22, y1 - 12 - k * sz
    vf = View(sh, shape, "vorne", k, (fx, fz), ref3d=(bb.min.X, 0, bb.min.Z), title=VIEW_TITLE["vorne"])
    vt = View(sh, shape, "oben", k, (fx, fz - gap - k * sy), ref3d=(bb.min.X, bb.min.Y, 0),
              title=VIEW_TITLE["oben"]) if "oben" in views else None
    vl = View(sh, shape, "links", k, (fx + k * sx + gap, fz), ref3d=(0, bb.max.Y, bb.min.Z),
              title=VIEW_TITLE["links"]) if "links" in views else None
    dim(sh, vf.pt((bb.min.X, 0, bb.max.Z)), vf.pt((bb.max.X, 0, bb.max.Z)), 6, mm(sx),
        register=("Länge X", sx))
    dim(sh, vf.pt((bb.min.X, 0, bb.min.Z)), vf.pt((bb.min.X, 0, bb.max.Z)), -6, mm(sz),
        horiz=False, register=("Höhe Z", sz))
    if vt:
        dim(sh, vt.pt((bb.max.X, bb.min.Y, 0)), vt.pt((bb.max.X, bb.max.Y, 0)), 6, mm(sy),
            horiz=False, register=("Breite Y", sy))
    hl = holes(shape)
    if hl:
        rows = []
        for i, h in enumerate(hl, 1):
            rows.append([i, mm(h["x"] - bb.min.X), mm(h["y"] - bb.min.Y), mm(h["z"] - bb.min.Z),
                         f"Ø{mm(h['d'])}"])
            sh.register(f"Bohrung {i}", [h["x"] - bb.min.X, h["y"] - bb.min.Y, h["z"] - bb.min.Z, h["d"]])
            for v in (vf, vt):
                if v:
                    q = v.pt((h["x"], h["y"], h["z"]))
                    sh.ax.text(q[0] + 1.5, q[1] + 1.5, str(i), fontsize=5, color="0.2")
        yb = table(sh, 300, y1, ["Nr.", "X", "Y", "Z", "Ø"], rows, [10, 22, 22, 22, 20], size=6)
        sh.note(300, yb - 4, ["Bohrungskoordinaten ab Bezugsecke", "(min X / min Y / min Z des Teils)."], 6)
    if notes:
        sh.note(x0 + 2, y0 + 30, list(notes), 6.5)
    return sh
