"""UR DH kinematics: forward kinematics and the analytic (Hawkins) inverse
with all 8 solutions.

DH parameters are Universal Robots' published values ("DH parameters for
calculations of kinematics and dynamics"), in mm. Link masses are from the
same page; `mass_kg` is the catalogue arm weight (the base housing gets the
rest). Frames: index 0 = robot base, 1..6 = DH frames, 6 = tool flange
(z out of the flange face). Angles in rad, zero = UR's zero position.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class URModel:
    name: str
    d1: float
    a2: float
    a3: float
    d4: float
    d5: float
    d6: float
    mass_kg: float
    link_masses: tuple[float, ...]   # links 1..6 (moving)
    payload_kg: float
    reach_mm: float
    alpha: tuple[float, ...] = field(default=(np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0))

    @property
    def a(self) -> np.ndarray:
        return np.array([0, self.a2, self.a3, 0, 0, 0], float)

    @property
    def d(self) -> np.ndarray:
        return np.array([self.d1, 0, 0, self.d4, self.d5, self.d6], float)

    @property
    def base_mass(self) -> float:
        """Base housing (does not move): catalogue weight minus links 1-6."""
        return self.mass_kg - sum(self.link_masses)

    @property
    def size(self) -> float:
        """Scale of the housings relative to the UR10e/UR16e (shared joint
        sizes), used to scale the capsule collision model."""
        return self.d4 / 174.15

    def as_dict(self) -> dict:
        return {"name": self.name, "dh_mm": {"d1": self.d1, "a2": self.a2, "a3": self.a3,
                                             "d4": self.d4, "d5": self.d5, "d6": self.d6},
                "mass_kg": self.mass_kg, "payload_kg": self.payload_kg,
                "reach_mm": self.reach_mm}


MODELS: dict[str, URModel] = {m.name: m for m in (
    URModel("UR3e", 151.85, -243.55, -213.2, 131.05, 85.35, 92.1, 11.2,
            (1.98, 3.4445, 1.437, 0.871, 0.805, 0.261), 3.0, 500),
    URModel("UR5e", 162.5, -425.0, -392.2, 133.3, 99.7, 99.6, 20.6,
            (3.761, 8.058, 2.846, 1.37, 1.3, 0.365), 5.0, 850),
    URModel("UR10e", 180.7, -612.7, -571.55, 174.15, 119.85, 116.55, 33.5,
            (7.369, 13.051, 3.989, 2.1, 1.98, 0.615), 12.5, 1300),
    # UR12e: the UR10e arm with a higher payload rating (same DH and STEP links)
    URModel("UR12e", 180.7, -612.7, -571.55, 174.15, 119.85, 116.55, 33.5,
            (7.369, 13.051, 3.989, 2.1, 1.98, 0.615), 12.5, 1300),
    URModel("UR16e", 180.7, -478.4, -360.0, 174.15, 119.85, 116.55, 33.1,
            (7.369, 10.45, 4.321, 2.18, 2.033, 0.907), 16.0, 900),
    URModel("UR20", 236.3, -862.0, -728.7, 201.0, 159.3, 154.3, 64.0,
            (16.343, 29.632, 7.879, 3.054, 3.126, 0.846), 20.0, 1750),
)}


def get_model(model: str | URModel) -> URModel:
    """'UR16e', 'ur16e', 'UR 16e' -> URModel."""
    if isinstance(model, URModel):
        return model
    key = str(model).replace(" ", "").replace("-", "").lower()
    for name, m in MODELS.items():
        if name.lower() == key:
            return m
    raise ValueError(f"Unknown UR model {model!r}; known: {sorted(MODELS)}")


def dh(model: URModel, theta: float, i: int) -> np.ndarray:
    """Standard DH transform frame i -> frame i+1 (i = 0..5)."""
    ct, st = np.cos(theta), np.sin(theta)
    al = model.alpha[i]
    ca, sa = np.cos(al), np.sin(al)
    a, d = model.a[i], model.d[i]
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0, sa, ca, d],
                     [0, 0, 0, 1.0]])


def fk_all(model: str | URModel, q) -> list[np.ndarray]:
    """[base, frame1 .. frame6 (flange)] as 4x4 in the robot base frame."""
    m = get_model(model)
    T = np.eye(4)
    out = [T]
    for i in range(6):
        T = T @ dh(m, q[i], i)
        out.append(T)
    return out


def fk(model: str | URModel, q) -> np.ndarray:
    """Flange pose (4x4) in the robot base frame."""
    return fk_all(model, q)[-1]


def wrap(q) -> np.ndarray:
    """Angles to [-pi, pi)."""
    return (np.asarray(q, float) + np.pi) % (2 * np.pi) - np.pi


def ik(model: str | URModel, T06, atol: float = 1e-3) -> list[np.ndarray]:
    """Analytic UR IK: flange pose in the base frame -> up to 8 joint
    vectors (wrapped to [-pi, pi)), each verified against FK. Wrist-singular
    poses (sin q5 = 0) and unreachable ones yield fewer or no solutions."""
    m = get_model(model)
    A2, A3, D4, D6 = m.a2, m.a3, m.d4, m.d6
    T06 = np.asarray(T06, float)
    sols: list[np.ndarray] = []
    p05 = T06 @ np.array([0, 0, -D6, 1])
    r = np.hypot(p05[0], p05[1])
    if r < abs(D4):
        return sols
    phi = np.arctan2(p05[1], p05[0])
    psi = np.arccos(D4 / r)
    T60 = np.linalg.inv(T06)
    X, Y = T60[:3, 0], T60[:3, 1]
    p = T06[:3, 3]
    for t1 in (phi + psi + np.pi / 2, phi - psi + np.pi / 2):
        s1, c1 = np.sin(t1), np.cos(t1)
        v = (p[0] * s1 - p[1] * c1 - D4) / D6
        if abs(v) > 1:
            continue
        for t5 in (np.arccos(v), -np.arccos(v)):
            s5 = np.sin(t5)
            if abs(s5) < 1e-6:
                continue
            t6 = np.arctan2((-X[1] * s1 + Y[1] * c1) / s5, (X[0] * s1 - Y[0] * c1) / s5)
            T14 = np.linalg.inv(dh(m, t1, 0)) @ T06 @ np.linalg.inv(dh(m, t5, 4) @ dh(m, t6, 5))
            P13 = T14 @ np.array([0, -D4, 0, 1]) - np.array([0, 0, 0, 1])
            n13 = np.linalg.norm(P13[:3])
            c3 = (n13 ** 2 - A2 ** 2 - A3 ** 2) / (2 * A2 * A3)
            if abs(c3) > 1:
                continue
            for t3 in (np.arccos(c3), -np.arccos(c3)):
                t2 = -np.arctan2(P13[1], -P13[0]) + np.arcsin(A3 * np.sin(t3) / n13)
                T34 = np.linalg.inv(dh(m, t2, 1) @ dh(m, t3, 2)) @ T14
                t4 = np.arctan2(T34[1, 0], T34[0, 0])
                q = wrap([t1, t2, t3, t4, t5, t6])
                if np.allclose(fk(m, q), T06, atol=atol):
                    sols.append(q)
    return sols


# ---- small pose helpers (JSON-friendly) -------------------------------------

def rpy_matrix(rpy_deg) -> np.ndarray:
    """Fixed-axis roll/pitch/yaw (deg) -> 3x3: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = np.radians(np.asarray(rpy_deg, float))
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def matrix_rpy(R) -> list[float]:
    """3x3 -> [roll, pitch, yaw] deg (inverse of rpy_matrix)."""
    R = np.asarray(R, float)
    p = np.arcsin(-np.clip(R[2, 0], -1, 1))
    if abs(np.cos(p)) > 1e-9:
        r, y = np.arctan2(R[2, 1], R[2, 2]), np.arctan2(R[1, 0], R[0, 0])
    else:
        r, y = np.arctan2(-R[1, 2], R[1, 1]), 0.0
    return [float(v) for v in np.degrees([r, p, y])]


def rotvec_matrix(rv) -> np.ndarray:
    """Axis-angle vector (rad, UR pendant 'RX RY RZ') -> 3x3."""
    rv = np.asarray(rv, float)
    th = np.linalg.norm(rv)
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def axis_angle(axis, angle: float) -> np.ndarray:
    """3x3 rotation about a unit axis (Rodrigues)."""
    ax = np.asarray(axis, float)
    return rotvec_matrix(ax / np.linalg.norm(ax) * angle)


def pose(spec) -> np.ndarray:
    """4x4 from a JSON-friendly spec: None (identity), a 4x4 nested list, or
    {"xyz": [mm], "rpy_deg": [r, p, y]} / {"xyz", "rotvec": [rad]} /
    {"xyz", "matrix": 3x3}."""
    if spec is None:
        return np.eye(4)
    if not isinstance(spec, dict):
        M = np.asarray(spec, float)
        if M.shape != (4, 4):
            raise ValueError(f"pose matrix must be 4x4, got {M.shape}")
        return M
    T = np.eye(4)
    T[:3, 3] = spec.get("xyz", (0, 0, 0))
    if "matrix" in spec:
        T[:3, :3] = np.asarray(spec["matrix"], float)
    elif "rotvec" in spec:
        T[:3, :3] = rotvec_matrix(spec["rotvec"])
    else:
        T[:3, :3] = rpy_matrix(spec.get("rpy_deg", (0, 0, 0)))
    return T


def pose_dict(T) -> dict:
    T = np.asarray(T, float)
    return {"xyz": [round(float(v), 3) for v in T[:3, 3]],
            "rpy_deg": [round(v, 4) for v in matrix_rpy(T[:3, :3])],
            "z_axis": [round(float(v), 5) for v in T[:3, 2]]}
