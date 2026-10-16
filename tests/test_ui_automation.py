"""Shapr3D UI automation: pure helpers plus the tools with macOS mocked."""

import sys
from pathlib import Path

import pytest

from shapr3d_mcp import macos, server

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS UI automation")

WIN = {"id": 1, "x": 0, "y": 34, "w": 1710, "h": 979}


def item(text, x, y):  # window points -> what read_ui returns
    return {"text": text, "x": x, "y": y, "screen_x": x, "screen_y": y + 34}


def test_to_points_flips_vision_origin():
    # Vision: normalized, bottom-left origin; box centred at (0.5, 0.25)
    p = macos.to_points({"text": "OK", "x": 0.4, "y": 0.2, "w": 0.2, "h": 0.1}, WIN)
    assert (p["x"], p["y"]) == (855, round(0.75 * 979))
    assert (p["screen_x"], p["screen_y"]) == (855, round(0.75 * 979) + 34)


def test_match_text_prefers_exact_then_orders_occurrences():
    items = [item("New Project", 855, 531), item("Current Project", 855, 497),
             item("Project", 100, 900), item("Project", 100, 50)]
    assert macos.match_text("current project", items)["y"] == 497
    assert macos.match_text("Project", items)["y"] == 50          # exact, topmost
    assert macos.match_text("Project", items, occurrence=2)["y"] == 900
    assert macos.match_text("Project", items, occurrence=3) is None
    assert macos.match_text("missing", items) is None


@pytest.mark.parametrize("combo,expected", [
    ("cmd+z", {"modifiers": ["command down"], "key": "z", "code": None}),
    ("escape", {"modifiers": [], "key": None, "code": 53}),
    ("cmd+shift+S", {"modifiers": ["command down", "shift down"], "key": "S", "code": None}),
])
def test_parse_keys(combo, expected):
    assert macos.parse_keys(combo) == expected


@pytest.mark.parametrize("bad", ["", "hyper+z", "cmd+notakey"])
def test_parse_keys_rejects(bad):
    with pytest.raises(ValueError):
        macos.parse_keys(bad)


@pytest.fixture
def fake_mac(monkeypatch):
    calls = {"clicks": [], "opens": []}
    monkeypatch.setattr(macos, "is_running", lambda: True)
    monkeypatch.setattr(macos, "activate", lambda: None)
    monkeypatch.setattr(macos, "window", lambda: WIN)
    monkeypatch.setattr(macos, "click_screen", lambda x, y: calls["clicks"].append((x, y)))
    monkeypatch.setattr(server.time, "sleep", lambda s: None)
    monkeypatch.setattr(server.subprocess, "run",
                        lambda *a, **k: calls["opens"].append(a[0]))
    return calls


DIALOG = [item("Import to", 855, 470), item("Where do you want to import", 855, 480),
          item("Current Project", 855, 497), item("New Project", 855, 531),
          item("Cancel", 855, 564)]


def test_click_text_not_found_lists_labels(fake_mac, monkeypatch):
    monkeypatch.setattr(macos, "read_ui", lambda out: (WIN, DIALOG))
    with pytest.raises(RuntimeError, match="Visible labels"):
        server.click_shapr3d_text("Export")
    assert fake_mac["clicks"] == []


def test_click_at_bounds_checked(fake_mac):
    server.click_shapr3d_at(10, 20)
    assert fake_mac["clicks"] == [(10, 54)]
    with pytest.raises(ValueError):
        server.click_shapr3d_at(5000, 20)


# ---------------------------------------------------------------------------
# export from Shapr3D + permissions warm-up
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_export(fake_mac, monkeypatch, tmp_path):
    calls = {"menu": [], "activated": [], "filled": []}
    monkeypatch.setattr(macos, "frontmost_app", lambda: "firefox")
    monkeypatch.setattr(macos, "activate_app", lambda n: calls["activated"].append(n))
    monkeypatch.setattr(macos, "menu", lambda p: calls["menu"].append(p))
    monkeypatch.setattr(macos, "wait_for_save_dialog", lambda t: True)

    def fill(name, folder):
        # Shapr3D writes the file without an extension when given a bare name
        from build123d import Box, export_step
        calls["filled"].append((name, folder))
        export_step(Box(3, 3, 3), str(Path(folder) / name))
    monkeypatch.setattr(macos, "fill_save_dialog", fill)
    return calls, tmp_path


def test_export_from_shapr3d_saves_adds_extension_restores_focus(fake_export):
    calls, folder = fake_export
    out = server.export_from_shapr3d("edited part", folder=str(folder))
    assert calls["menu"] == [["File", "Export to", "STEP…"]]
    assert calls["filled"] == [("edited part", str(folder))]
    assert calls["activated"] == ["firefox"]          # focus handed back
    assert out["saved"] == str(folder / "edited part.step")
    assert out["summary"]["solids"] == 1


def test_export_without_dialog_reports_labels(fake_export, monkeypatch):
    monkeypatch.setattr(macos, "wait_for_save_dialog", lambda t: False)
    monkeypatch.setattr(macos, "read_ui", lambda out: (WIN, [item("Export Options", 800, 400)]))
    with pytest.raises(RuntimeError, match="Export Options"):
        server.export_from_shapr3d("x", folder=str(fake_export[1]))


def test_export_rejects_unknown_format(fake_mac):
    with pytest.raises(ValueError, match="fmt must be"):
        server.export_from_shapr3d("x", fmt="dwg")


def test_check_permissions_reports_missing(monkeypatch):
    monkeypatch.setattr(macos, "permissions", lambda prompt: {
        "accessibility": True, "screen_recording": False,
        "automation_system_events": True, "error": ""})
    monkeypatch.setattr(macos, "is_running", lambda: False)
    out = server.check_permissions(prompt=False)
    assert out["ready"] is False and "screen_recording" in out["todo"]
    assert "mcp__shapr3d" in out["claude_tool_approvals"]


def test_keys_types_text(fake_mac, monkeypatch):
    typed = []
    monkeypatch.setattr(macos, "type_text", typed.append)
    monkeypatch.setattr(macos, "keys", lambda c: typed.append(f"<{c}>"))
    server.shapr3d_keys(combo="return", text="12.5")
    assert typed == ["12.5", "<return>"]
    with pytest.raises(ValueError):
        server.shapr3d_keys()


# ---------------------------------------------------------------------------
# open_in_shapr3d: iterate in the open project unless asked for a new one
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_import(monkeypatch):
    scripts = []
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(server.time, "sleep", lambda s: None)
    monkeypatch.setattr(server, "_osascript", lambda sc: scripts.append(sc) or "")
    monkeypatch.setattr(server, "_shapr3d_history_mentions", lambda needle: True)
    server.create_model("iterate_me", "result = Box(5, 5, 5)")
    return scripts


def _menu(items):
    return lambda path: items


def test_open_imports_into_open_project_by_default(fake_import, monkeypatch):
    monkeypatch.setattr(macos, "menu_items", _menu(["New Project", "Import File…", "Export to"]))
    msg = server.open_in_shapr3d("iterate_me.step")
    assert 'keystroke "n"' not in fake_import[0] and "the current" in msg


def test_open_creates_project_from_home_screen(fake_import, monkeypatch):
    monkeypatch.setattr(macos, "menu_items", _menu(["New Project", "Import as new project…"]))
    msg = server.open_in_shapr3d("iterate_me.step")
    assert 'keystroke "n"' in fake_import[0] and "a new" in msg


def test_open_new_project_only_when_asked(fake_import, monkeypatch):
    monkeypatch.setattr(macos, "menu_items", _menu(["Import File…"]))
    server.open_in_shapr3d("iterate_me.step", new_project=True)
    assert 'keystroke "n"' in fake_import[0]


def test_unreadable_menu_never_starts_new_project(fake_import, monkeypatch):
    def boom(path):
        raise RuntimeError("no access")
    monkeypatch.setattr(macos, "menu_items", boom)
    server.open_in_shapr3d("iterate_me.step")
    assert 'keystroke "n"' not in fake_import[0]
