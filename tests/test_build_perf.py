"""Fast paths of the build report: mesh-based bboxes, shared volumes,
z-buffer renders."""

from build123d import Box, Compound, Cylinder, Pos, Rot, Sphere

from shapr3d_mcp import kernel, server


def test_fast_bbox_mesh_path_close_to_exact(monkeypatch):
    shape = Compound(children=[Pos(3, 4, 5) * Sphere(7), Rot(30, 0, 0) * Cylinder(5, 10)])
    exact = shape.bounding_box()
    monkeypatch.setattr(kernel, "EXACT_BBOX_MAX_FACES", 0)
    fast = kernel.fast_bbox(shape)
    for a, b in zip((*exact.min, *exact.max), (*fast.min, *fast.max)):
        assert abs(a - b) < 0.1
    # the measuring mesh is dropped again (later STL exports pick their own)
    from OCP.BRepTools import BRepTools
    assert not BRepTools.Triangulation_s(shape.wrapped, 1.0)


def test_fast_bbox_small_shapes_stay_exact():
    bb = kernel.fast_bbox(Pos(1, 2, 3) * Cylinder(5, 10))
    assert [round(v, 6) for v in (*bb.min, *bb.max)] == [-4, -3, -2, 6, 7, 8]


def test_volume_memo_shares_moved_copies(monkeypatch):
    base = Box(10, 20, 30) - Cylinder(3, 40)
    copies = [Pos(x, 0, 0) * base for x in (0, 100, 200)] + [Rot(0, 90, 0) * base]
    calls = []
    orig = type(base).volume
    monkeypatch.setattr(type(base), "volume",
                        property(lambda s: calls.append(1) or orig.fget(s)))
    memo = kernel.Measures()
    total = memo.total(copies)
    assert len(calls) == 1
    assert abs(total - 4 * orig.fget(base)) < 1e-6
    other = Box(1, 2, 3)
    assert abs(memo.volume(other) - 6) < 1e-9 and len(calls) == 2


def test_rasterizer_z_test_and_background():
    import numpy as np
    from shapr3d_mcp import raster

    # front view (camera at -Y): red square in front of a wider blue one
    def quad(y, x0, x1):
        a, b, c, d = [x0, y, 0], [x1, y, 0], [x1, y, 10], [x0, y, 10]
        return [[a, b, c], [a, c, d]]
    tris = np.array(quad(-5, 0, 10) + quad(5, -5, 15), dtype=np.float32)
    rgb = np.array([[1, 0, 0]] * 2 + [[0, 0, 1]] * 2, dtype=np.float32)
    for ortho in (True, False):
        cam = raster.Camera(0, -90, ortho, np.array([5.0, 0, 5]), 10)
        canvas = raster.Canvas(40, 40, 1.5, 0, 0)
        canvas.draw(tris, raster.shade(tris, rgb, cam), cam)
        img = canvas.image()
        assert img[20, 20][0] > 0.5 and img[20, 20][2] < 0.05  # red wins
        assert img[20, 8][2] > 0.3 and img[20, 8][0] < 0.05    # blue beside it
        assert (img[0, 0] == 1).all()                          # white background


def test_render_png_views_and_edges(tmp_path):
    out = tmp_path / "r.png"
    server._render_png(Sphere(20), out, [("iso", 30, -60, False), ("top", 90, -90, True)],
                       300, 200, True)
    assert out.stat().st_size > 1000


def test_shape_cache_replaces_stale_version(tmp_path, monkeypatch):
    import os
    from build123d import export_step

    monkeypatch.setattr(server, "_SHAPE_CACHE", type(server._SHAPE_CACHE)())
    f = tmp_path / "a.step"
    export_step(Box(1, 1, 1), str(f))
    server._import_cached(f)
    export_step(Box(2, 2, 2), str(f))
    os.utime(f, ns=(1, 1))  # new mtime -> new cache key
    shape = server._import_cached(f)
    assert len(server._SHAPE_CACHE) == 1
    assert abs(shape.volume - 8) < 1e-6


def _fake_step(path, size=1):
    from build123d import export_step
    export_step(Box(size, size, size), str(path))


def test_load_cad_drops_stale_versions(tmp_path):
    import os
    from build123d import import_step

    cache = tmp_path / ".cache"
    src = tmp_path / "asm.step"
    other = tmp_path / "asm-b.step"      # stem shares the prefix: must survive
    _fake_step(other)
    kernel.load_cad(other, lambda p: import_step(str(p)), cache)
    for i in range(3):
        _fake_step(src, i + 1)
        os.utime(src, ns=(i + 1, i + 1))
        kernel.load_cad(src, lambda p: import_step(str(p)), cache)
    names = sorted(f.name for f in cache.iterdir())
    assert len(names) == 2
    assert any(n.startswith("asm-b-") for n in names)
    assert f"asm-{kernel.cache_key(src)}.brep" in names


def test_prune_cache_age_and_size(tmp_path):
    import os
    import time

    cache = tmp_path / ".cache"
    cache.mkdir()
    now = time.time()
    for i, age_days in enumerate((40, 3, 2, 1)):
        f = cache / f"p{i}-{i:016x}.brep"
        f.write_bytes(b"x" * 1000)
        os.utime(f, (now - age_days * 86400,) * 2)
    removed = kernel.prune_cache(cache, max_bytes=2500)
    assert sorted(f.name[:2] for f in removed) == ["p0", "p1"]  # too old, then LRU
    assert sorted(f.name[:2] for f in cache.iterdir()) == ["p2", "p3"]
