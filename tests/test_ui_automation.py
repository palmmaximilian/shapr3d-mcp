"""Shapr3D UI automation: pure helpers plus the tools with macOS mocked."""

import pytest

from shapr3d_mcp import macos, server

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


def test_open_answers_import_dialog(fake_mac, monkeypatch):
    server.create_model("ui_open", "result = Box(5, 5, 5)")
    screens = iter([[item("Modeling", 200, 60)], DIALOG])
    monkeypatch.setattr(macos, "read_ui", lambda out: (WIN, next(screens)))
    out = server.open_in_shapr3d("ui_open.step", import_to="new")
    assert out["import_to"] == "New Project"
    assert fake_mac["clicks"] == [(855, 531 + 34)]
    assert fake_mac["opens"][0][:3] == ["open", "-a", "Shapr3D"]


def test_open_without_dialog_reports_instead_of_clicking(fake_mac, monkeypatch):
    server.create_model("ui_open2", "result = Box(5, 5, 5)")
    monkeypatch.setattr(macos, "read_ui", lambda out: (WIN, [item("Modeling", 200, 60)]))
    out = server.open_in_shapr3d("ui_open2.step", import_to="current", timeout_s=0.01)
    assert out["import_to"] is None and fake_mac["clicks"] == []


def test_open_rejects_bad_choice(fake_mac):
    server.create_model("ui_open3", "result = Box(5, 5, 5)")
    with pytest.raises(ValueError):
        server.open_in_shapr3d("ui_open3.step", import_to="sideways")


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
