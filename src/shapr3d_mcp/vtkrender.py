"""Colour renderer (VTK, offscreen): one colour per part group, lighting and
dark feature-edge outlines.

Runs as its own process (`python -m shapr3d_mcp.vtkrender spec.json`): on
macOS a VTK render window needs the main thread, and MCP tools run in job
threads. The server meshes the shapes (OCC) and hands over plain arrays:

spec.json  {"arrays": npz path, "colors": [[r, g, b]...] per group,
            "views": [{"elev", "azim", "ortho", "width", "height", "out"}],
            "outlines": bool, "edges": bool, "line_width": px}
npz        t0, t1... float32 (N, 3, 3) triangles per group;
           optional edge_pts (M, 3) + edge_offsets (K + 1,) polylines.
"""

from __future__ import annotations

import json
import sys

import numpy as np

OUTLINE_RGB = (0.12, 0.12, 0.14)
EDGE_RGB = (0.07, 0.07, 0.07)
MARGIN = 1.04  # fit slack around the projected bounding box


def _polydata(tris: np.ndarray):
    """(N, 3, 3) triangles -> cleaned vtkPolyData with split normals."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    n = len(tris)
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(tris.reshape(-1, 3), np.float32), deep=True))
    cells = np.empty((n, 4), np.int64)
    cells[:, 0] = 3
    cells[:, 1:] = np.arange(3 * n, dtype=np.int64).reshape(n, 3)
    ca = vtk.vtkCellArray()
    ca.SetCells(n, numpy_to_vtkIdTypeArray(cells.ravel(), deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(ca)
    clean = vtk.vtkCleanPolyData()  # STL soup -> shared vertices
    clean.SetInputData(pd)
    clean.Update()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(clean.GetOutput())
    normals.SetFeatureAngle(35)
    normals.SplittingOn()
    normals.ConsistencyOn()
    normals.Update()
    return clean.GetOutput(), normals.GetOutput()


def _outline_actor(pd, line_width: float):
    import vtk

    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(pd)
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOn()
    fe.SetFeatureAngle(40)
    fe.ManifoldEdgesOff()
    fe.NonManifoldEdgesOff()
    fe.Update()
    return _line_actor(fe.GetOutput(), OUTLINE_RGB, line_width)


def _line_actor(pd, rgb, line_width: float):
    import vtk

    m = vtk.vtkPolyDataMapper()
    m.SetInputData(pd)
    m.ScalarVisibilityOff()
    a = vtk.vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetColor(*rgb)
    a.GetProperty().SetLineWidth(line_width)
    a.GetProperty().LightingOff()
    return a


def _edge_polydata(pts: np.ndarray, offsets: np.ndarray):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    vp = vtk.vtkPoints()
    vp.SetData(numpy_to_vtk(np.ascontiguousarray(pts, np.float32), deep=True))
    lines = vtk.vtkCellArray()
    for a, b in zip(offsets[:-1], offsets[1:]):
        lines.InsertNextCell(int(b - a))
        for i in range(int(a), int(b)):
            lines.InsertCellPoint(i)
    pd = vtk.vtkPolyData()
    pd.SetPoints(vp)
    pd.SetLines(lines)
    return pd


def camera_axes(elev: float, azim: float):
    """matplotlib view_init convention (as raster.Camera): direction from
    the model towards the camera, screen right and screen up."""
    e, a = np.radians(elev), np.radians(azim)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    return d, right, np.cross(d, right)


def _place_camera(cam, view: dict, lo: np.ndarray, hi: np.ndarray) -> None:
    """Point the camera at the box centre and fit the box to the image."""
    center = (lo + hi) / 2
    radius = max(float(np.linalg.norm(hi - lo)) / 2, 1e-9)
    d, right, up = camera_axes(view["elev"], view["azim"])
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) - center
    x, y, depth = corners @ right, corners @ up, corners @ d
    aspect = view["width"] / view["height"]
    dist = 4.0 * radius  # mild perspective, as the raster renderer
    cam.SetFocalPoint(*center)
    cam.SetPosition(*(center + dist * d))
    cam.SetViewUp(*up)
    if view["ortho"]:
        cam.ParallelProjectionOn()
        cam.SetParallelScale(MARGIN * max(np.abs(y).max(), np.abs(x).max() / aspect))
    else:
        f = dist / np.maximum(dist - depth, 1e-6 * dist)
        half = MARGIN * max(np.abs(y * f).max(), np.abs(x * f).max() / aspect)
        cam.SetViewAngle(float(np.degrees(2 * np.arctan(half / dist))))
    cam.SetClippingRange(max(dist - 2 * radius, 1e-3 * dist), dist + 2 * radius)


def render(spec: dict) -> None:
    import vtk

    data = np.load(spec["arrays"])
    groups = []
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for i, rgb in enumerate(spec["colors"]):
        t = data[f"t{i}"]
        if not len(t):
            continue
        p = t.reshape(-1, 3)
        lo, hi = np.minimum(lo, p.min(axis=0)), np.maximum(hi, p.max(axis=0))
        groups.append((*_polydata(t), rgb))
    edge_pd = None
    if spec.get("edges") and "edge_pts" in data and len(data["edge_pts"]):
        edge_pd = _edge_polydata(data["edge_pts"], data["edge_offsets"])

    lw = float(spec.get("line_width", 1.0))
    for view in spec["views"]:
        ren = vtk.vtkRenderer()
        ren.SetBackground(1, 1, 1)
        ren.SetLayer(0)
        for clean, shaded, rgb in groups:
            m = vtk.vtkPolyDataMapper()
            m.SetInputData(shaded)
            a = vtk.vtkActor()
            a.SetMapper(m)
            p = a.GetProperty()
            p.SetColor(*rgb)
            p.SetAmbient(0.45)
            p.SetDiffuse(0.65)
            p.SetSpecular(0.15)
            p.SetSpecularPower(20)
            ren.AddActor(a)
            if spec.get("outlines", True):
                ren.AddActor(_outline_actor(clean, lw))
        _place_camera(ren.GetActiveCamera(), view, lo, hi)
        light = vtk.vtkLight()
        light.SetLightTypeToCameraLight()
        light.SetIntensity(0.55)
        ren.AddLight(light)

        win = vtk.vtkRenderWindow()
        win.SetOffScreenRendering(1)
        win.SetSize(int(view["width"]), int(view["height"]))
        win.SetMultiSamples(8)
        win.AddRenderer(ren)
        if edge_pd is not None:  # wireframe on top: hidden edges show too
            win.SetNumberOfLayers(2)
            top = vtk.vtkRenderer()
            top.SetLayer(1)
            top.InteractiveOff()
            top.SetActiveCamera(ren.GetActiveCamera())
            top.AddActor(_line_actor(edge_pd, EDGE_RGB, lw))
            win.AddRenderer(top)
        win.Render()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(win)
        w2i.ReadFrontBufferOff()
        w2i.Update()
        wr = vtk.vtkPNGWriter()
        wr.SetFileName(view["out"])
        wr.SetInputConnection(w2i.GetOutputPort())
        wr.Write()
        win.Finalize()


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as f:
        render(json.load(f))


if __name__ == "__main__":
    main()
