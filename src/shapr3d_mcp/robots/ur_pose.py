"""Pose the vendor UR STEP (universal-robots.com download) with DH angles.

Vendor UR STEP convention (checked on UR16e and UR12e/UR10e): Y up, base axis
= Y, shoulder offset towards +Z, the cable outlet towards -Z, arm in the
straight-up home pose (flange face at Y = d1 - a2 - a3 + d5, Z = d4 + d6).
The DH base frame is Rx(+90) of the STEP frame: x_b = X, y_b = -Z, z_b = Y.

The STEP has one top-level child per link (labels like "C-1000487"); which
child belongs to which DH frame is detected from where it sits in the home
pose, or passed in explicitly as {label: frame index 0..6}.
"""

from __future__ import annotations

import functools
import itertools

import numpy as np

from .ur_kinematics import URModel, fk_all, get_model

R_S2B = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])

LINK_NAMES = ("base", "shoulder", "upper arm", "forearm", "wrist 1", "wrist 2", "wrist 3")


def home_joint_origins(model: str | URModel) -> list[np.ndarray]:
    """DH frame 1..6 origins of the STEP home pose, in STEP coordinates."""
    m = get_model(model)
    h = m.d1 - m.a2 - m.a3
    return [np.array(p, float) for p in
            ([0, m.d1, 0], [0, m.d1 - m.a2, 0], [0, h, 0], [0, h, m.d4],
             [0, h + m.d5, m.d4], [0, h + m.d5, m.d4 + m.d6])]


@functools.lru_cache(maxsize=None)
def _home_q(name: str, s2b: bytes) -> tuple[float, ...]:
    S2B = np.frombuffer(s2b).reshape(4, 4)
    want = [S2B[:3, :3] @ p for p in home_joint_origins(name)]
    zf = S2B[:3, :3] @ np.array([0, 0, 1.0])
    best = None
    # the wrist is singular there (q5 = 0): brute-force quarter turns and
    # match every joint origin + the flange axis
    for q in itertools.product(np.radians([0, 90, 180, -90]), repeat=6):
        fr = fk_all(name, np.array(q))
        err = sum(np.linalg.norm(fr[i + 1][:3, 3] - w) for i, w in enumerate(want))
        err += 100 * np.linalg.norm(fr[6][:3, 2] - zf)
        if best is None or err < best[0]:
            best = (err, q)
    if best[0] > 5:
        raise ValueError(f"no DH pose matches the {name} STEP home pose (err {best[0]:.1f})")
    return best[1]


def home_q(model: str | URModel, step_to_base=R_S2B) -> np.ndarray:
    """DH angles of the vendor STEP's home pose."""
    m = get_model(model)
    return np.array(_home_q(m.name, np.ascontiguousarray(step_to_base, float).tobytes()))


def link_reference_points(model: str | URModel) -> np.ndarray:
    """Typical centre of each link's housing in the STEP home pose (7 x 3)."""
    m = get_model(model)
    k = m.size
    h = m.d1 - m.a2 - m.a3
    return np.array([[0, 0.25 * m.d1, 0], [0, m.d1, 0],
                     [0, m.d1 - m.a2 / 2, 176 * k], [0, m.d1 - m.a2 - m.a3 / 2, 40 * k],
                     [0, h, m.d4], [0, h + 0.9 * m.d5, m.d4], [0, h + m.d5, m.d4 + 0.8 * m.d6]])


def detect_link_map(robot, model: str | URModel, step_to_base=R_S2B) -> list[int]:
    """DH frame index (0..6) for each top-level child of the vendor STEP,
    by the nearest link reference point in the home pose. Raises if a link
    gets no part (not a UR vendor STEP in home pose, or a different model)."""
    m = get_model(model)
    S2B = np.asarray(step_to_base, float)
    # reference points live in the vendor frame; for a custom step_to_base
    # map them through base coordinates
    refs = (np.linalg.inv(S2B) @ R_S2B @ np.c_[link_reference_points(m), np.ones(7)].T).T[:, :3]
    children = list(robot.children)
    if not children:
        raise ValueError("robot STEP has no assembly children (one per link expected)")
    out = []
    for c in children:
        cc = c.bounding_box().center()
        p = np.array([cc.X, cc.Y, cc.Z])
        out.append(int(np.argmin(np.linalg.norm(refs - p, axis=1))))
    missing = sorted(set(range(7)) - set(out))
    if missing:
        raise ValueError(
            f"could not match STEP children to {m.name} links "
            f"{[LINK_NAMES[i] for i in missing]}; pass link_map={{label: frame index}}. "
            f"Children: {[c.label for c in children]}")
    return out


def to_loc(M):
    """4x4 (mm) -> build123d Location."""
    from build123d import Location
    from OCP.gp import gp_Trsf
    from OCP.TopLoc import TopLoc_Location

    M = np.asarray(M, float)
    t = gp_Trsf()
    t.SetValues(*M[0, :4], *M[1, :4], *M[2, :4])
    return Location(TopLoc_Location(t))


def posed_robot(robot, model: str | URModel, q, Tbase=None, link_map: dict | None = None,
                step_to_base=R_S2B, label_prefix: str | None = None) -> list:
    """Moved copies of the vendor STEP's link children for joint angles q
    (rad) and base pose Tbase (4x4 world <- robot base), base link first.
    link_map: {child label: DH frame index}; None = detect."""
    m = get_model(model)
    S2B = np.asarray(step_to_base, float)
    Tbase = np.eye(4) if Tbase is None else np.asarray(Tbase, float)
    children = list(robot.children)
    if link_map is None:
        idx = detect_link_map(robot, m, S2B)
    else:
        idx = [link_map.get(c.label) for c in children]
    F0, F = fk_all(m, home_q(m, S2B)), fk_all(m, q)
    prefix = m.name if label_prefix is None else label_prefix
    parts = []
    for c, i in sorted(zip(children, idx), key=lambda ci: (ci[1] is None, ci[1] or 0)):
        if i is None:
            continue
        M = Tbase @ F[i] @ np.linalg.inv(F0[i]) @ S2B
        moved = c.moved(to_loc(M))
        moved.label = f"{prefix} {c.label}".strip()
        parts.append(moved)
    return parts


def flange_pose(model: str | URModel, q, Tbase=None) -> np.ndarray:
    Tbase = np.eye(4) if Tbase is None else np.asarray(Tbase, float)
    return Tbase @ fk_all(model, q)[-1]
