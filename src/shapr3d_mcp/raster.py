"""Numpy z-buffer rasterizer for preview renders.

Flat-shaded triangles, one colour per triangle, no GPU or display. Memory
stays bounded by processing triangles in chunks of candidate pixels, so
vendor CAD with millions of triangles renders in seconds at a few hundred
MB (matplotlib's Poly3DCollection held ~2-3 kB per triangle and view).
"""

from __future__ import annotations

import numpy as np

SUPERSAMPLE = 2           # render at 2x and box-filter down (antialiasing)
CHUNK_FRAGMENTS = 500_000  # candidate pixels tested per numpy pass
BACKGROUND = 1.0          # white


class Camera:
    """matplotlib view_init convention: camera on the unit sphere at
    (elevation, azimuth) looking at the model centre, +Z up."""

    def __init__(self, elev: float, azim: float, ortho: bool,
                 center: np.ndarray, radius: float):
        e, a = np.radians(elev), np.radians(azim)
        self.dir = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        self.right = np.array([-np.sin(a), np.cos(a), 0.0])
        self.up = np.cross(self.dir, self.right)
        self.ortho = ortho
        self.center = center
        # perspective eye distance: mild foreshortening, close to mplot3d's
        self.eye_dist = 4.0 * max(radius, 1e-9)
        # Headlight from the camera, nudged up-left so faces square to the
        # view still differ from their neighbours.
        light = self.dir + 0.35 * np.array([-np.sin(a), np.cos(a), 0.6])
        self.light = light / np.linalg.norm(light)

    def project(self, pts: np.ndarray):
        """World points (..., 3) -> (screen x, screen y, depth), screen in
        model units (y up), depth larger = closer. Perspective depth is
        1/distance, which interpolates linearly in screen space."""
        dt = pts.dtype  # float32 meshes stay float32 (half the memory)
        rel = pts - self.center.astype(dt)
        x = rel @ self.right.astype(dt)
        y = rel @ self.up.astype(dt)
        d = rel @ self.dir.astype(dt)
        if self.ortho:
            return x, y, d
        dist = np.maximum(self.eye_dist - d, 1e-6 * self.eye_dist)
        f = self.eye_dist / dist
        return x * f, y * f, 1.0 / dist


def shade(tris: np.ndarray, base_rgb: np.ndarray, cam: Camera) -> np.ndarray:
    """Per-triangle colour: 0.3 ambient + 0.7 Lambert from the headlight.
    Normals face the camera (two-sided), so open shells shade too."""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    n *= np.where(n @ cam.dir < 0, -1.0, 1.0)[:, None]
    s = 0.3 + 0.7 * np.clip(n @ cam.light, 0, 1)
    return (base_rgb * s[:, None]).astype(np.float32)


class Canvas:
    """Z-buffer at SUPERSAMPLE x the output size. `to_px` maps screen
    units to (supersampled) pixel coordinates."""

    def __init__(self, width: int, height: int, scale: float, cx: float, cy: float):
        self.w, self.h = width * SUPERSAMPLE, height * SUPERSAMPLE
        # python floats, so float32 coordinates stay float32 (NEP 50)
        self.scale = float(scale) * SUPERSAMPLE
        self.cx, self.cy = float(cx), float(cy)
        self.z = np.full(self.w * self.h, -np.inf, dtype=np.float64)
        self.rgb = np.full((self.w * self.h, 3), BACKGROUND, dtype=np.float32)

    def to_px(self, x, y):
        return ((x - self.cx) * self.scale + self.w / 2,
                self.h / 2 - (y - self.cy) * self.scale)

    def draw(self, tris: np.ndarray, colors: np.ndarray, cam: Camera) -> None:
        """Rasterize (N, 3, 3) world triangles with (N, 3) colours."""
        if not len(tris):
            return
        sx, sy, sz = cam.project(tris)
        px, py = self.to_px(sx, sy)
        # pixel centres sit at i + 0.5: candidate columns/rows per triangle
        x0 = np.maximum(np.ceil(px.min(axis=1) - 0.5), 0).astype(np.int32)
        x1 = np.minimum(np.floor(px.max(axis=1) - 0.5), self.w - 1).astype(np.int32)
        y0 = np.maximum(np.ceil(py.min(axis=1) - 0.5), 0).astype(np.int32)
        y1 = np.minimum(np.floor(py.max(axis=1) - 0.5), self.h - 1).astype(np.int32)
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        area = ((px[:, 1] - px[:, 0]) * (py[:, 2] - py[:, 0])
                - (px[:, 2] - px[:, 0]) * (py[:, 1] - py[:, 0]))
        keep = (bw > 0) & (bh > 0) & (np.abs(area) > 1e-12)
        idx = np.nonzero(keep)[0]
        if not len(idx):
            return
        n = bw[idx].astype(np.int64) * bh[idx]
        # bucket by candidate count (1, 4, 16, ...): each bucket tests a
        # fixed-size block of k pixels per triangle, masked to its bbox
        lo = 0
        k = 1
        while lo < n.max():
            sel = idx[(n > lo) & (n <= k)]
            if len(sel):
                step = max(1, CHUNK_FRAGMENTS // k)
                for s in range(0, len(sel), step):
                    self._fill(sel[s:s + step], k, px, py, sz, area, x0, y0, bw, bh, colors)
            lo, k = k, k * 4

    def _fill(self, t, k, px, py, sz, area, x0, y0, bw, bh, colors) -> None:
        off = np.arange(k, dtype=np.int64)
        w = bw[t][:, None]
        valid = off[None, :] < (w * bh[t][:, None])
        ti, oi = np.nonzero(valid)
        tri = t[ti]
        wi = bw[tri]
        cx = x0[tri] + oi % wi
        cy = y0[tri] + oi // wi
        fx, fy = cx + 0.5, cy + 0.5
        X, Y, A = px[tri], py[tri], area[tri]
        # barycentric weights from edge functions; inclusive edges (small
        # tolerance) so neighbouring triangles leave no cracks
        b0 = ((X[:, 1] - fx) * (Y[:, 2] - fy) - (X[:, 2] - fx) * (Y[:, 1] - fy)) / A
        b1 = ((X[:, 2] - fx) * (Y[:, 0] - fy) - (X[:, 0] - fx) * (Y[:, 2] - fy)) / A
        b2 = 1.0 - b0 - b1
        inside = (b0 >= -1e-7) & (b1 >= -1e-7) & (b2 >= -1e-7)
        if not inside.any():
            return
        tri, b0, b1, b2 = tri[inside], b0[inside], b1[inside], b2[inside]
        pix = cy[inside] * self.w + cx[inside]
        Z = sz[tri]
        depth = b0 * Z[:, 0] + b1 * Z[:, 1] + b2 * Z[:, 2]
        # z-test: keep the closest fragment per pixel (ties: any one)
        np.maximum.at(self.z, pix, depth)
        win = depth >= self.z[pix]
        self.rgb[pix[win]] = colors[tri[win]]

    def image(self) -> np.ndarray:
        """(h, w, 3) float image, box-filtered down by SUPERSAMPLE."""
        s = SUPERSAMPLE
        img = self.rgb.reshape(self.h // s, s, self.w // s, s, 3)
        return img.mean(axis=(1, 3))
