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

@pytest.fixture(autouse=True)
def fresh_import_manifest(tmp_path, monkeypatch):
    """Import versions are counted per model in a manifest: start each test clean."""
    monkeypatch.setattr(server, "IMPORTS_MANIFEST", tmp_path / "imports.json")


@pytest.fixture
def fake_import(monkeypatch):
    scripts = []
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(server.time, "sleep", lambda s: None)
    monkeypatch.setattr(server, "_osascript", lambda sc: scripts.append(sc) or "")
    monkeypatch.setattr(server, "_shapr3d_history_mentions", lambda needle: True)
    monkeypatch.setattr(macos, "read_ui", lambda out: (WIN, []))  # no sidebar
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



# ---------------------------------------------------------------------------
# replace the previous version on re-import
# ---------------------------------------------------------------------------

def row(text, y, left=60):
    return {"text": text, "x": left + 50, "y": y, "left": left,
            "screen_x": left + 50, "screen_y": y + 34}


SIDEBAR = [row("Items", 82, 24), row("9 All Items", 118, 30)]


@pytest.mark.parametrize("text,stem,hit", [
    ("› O vgc10_assembl...", "vgc10_assembly", True),        # truncated prefix
    (") vgc10_assembly", "vgc10_assembly", True),            # OCR glyph noise
    ("vgc10_assembly_step", "vgc10_assembly", True),
    ("› O vgc10_assembl...", "bracket", False),
    ("› O vgc...", "vgc10_assembly", False),                 # prefix too short
    ("vgc10_assembly_edited", "vgc10_assembly", False),      # full other name
])
def test_name_matches(text, stem, hit):
    r = dict(row(text, 150))
    r["name"], r["truncated"] = macos.clean_item_name(text)
    assert macos.name_matches(r, stem) is hit


def test_item_rows_names_and_glyph_fragments():
    items = SIDEBAR + [row("› O vgc10_assembl...", 147), row("› 0", 171, left=27),
                       row("vgc10_assembl...", 172, left=70), row("Modeling", 82, 270)]
    rows = macos.item_rows(items)
    assert [(r["name"], r["y"]) for r in rows] == [("vgc10_assembl", 147), ("vgc10_assembl", 172)]


def test_all_items_anchor_without_space():
    items = [row("Items", 82, 24), row("AllItems", 119, 30), row("› O part", 147)]
    assert [r["name"] for r in macos.item_rows(items)] == ["part"]


def test_activate_waits_for_front(monkeypatch):
    fronts = iter(["firefox", "firefox", "Shapr3D"])
    monkeypatch.setattr(macos.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(macos.time, "sleep", lambda s: None)
    monkeypatch.setattr(macos, "_jxa", lambda script, *a, **k: next(fronts))
    macos.activate()  # returns only once Shapr3D is in front


def test_activate_refuses_when_never_front(monkeypatch):
    monkeypatch.setattr(macos.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(macos.time, "sleep", lambda s: None)
    monkeypatch.setattr(macos, "_jxa", lambda script, *a, **k: "firefox")
    with pytest.raises(RuntimeError, match="not clicking"):
        macos.activate(timeout_s=0)


def test_sidebar_edge_follows_resized_sidebar():
    narrow = [row("Modeling", 82, 270)]
    wide = [row("Modeling", 82, 385)]
    assert macos.sidebar_max_x(narrow) == 230 and macos.sidebar_max_x(wide) == 345
    assert macos.sidebar_max_x([]) == macos.SIDEBAR_MAX_X


def test_selected_body_count():
    assert macos.selected_body_count([row("Deselect All", 312), row("113 bodies", 328)]) == 113
    assert macos.selected_body_count([row("1 body", 328)]) == 1
    assert macos.selected_body_count([row("Modeling", 82)]) is None


@pytest.fixture
def screens(fake_import, monkeypatch):
    """Feed read_ui a scripted sequence of screens; record clicks/keys."""
    state = {"screens": [], "clicks": [], "keys": []}
    def read_ui(out):
        # scripted screens; the last one stays on screen
        scr = state["screens"]
        return WIN, (scr.pop(0) if len(scr) > 1 else scr[0])
    monkeypatch.setattr(macos, "read_ui", read_ui)
    monkeypatch.setattr(server, "_shapr3d_history_mentions",
                        lambda needle: pytest.fail("sidebar should confirm, not History"))
    monkeypatch.setattr(macos, "activate", lambda: None)
    monkeypatch.setattr(macos, "click_screen", lambda x, y: state["clicks"].append((x, y)))
    monkeypatch.setattr(macos, "keys", lambda c: state["keys"].append(c))
    monkeypatch.setattr(macos, "menu_items", lambda p: ["Import File…"])
    return state


OLD = row("› O iterate_me", 147)
SELECTED = SIDEBAR + [OLD, row("Deselect All", 312, 270), row("1 body", 328, 270),
                      row("Delete", 779, 270)]


NEW = row("› O iterate_me", 147)


def test_reimport_deletes_previous_version(screens, fake_import):
    # replace: list -> selected -> deleted; confirm: before (0) -> arrived (1)
    screens["screens"] = [SIDEBAR + [OLD], SELECTED, SIDEBAR, SIDEBAR, SIDEBAR + [NEW]]
    msg = server.open_in_shapr3d("iterate_me.step", versioned=False)
    assert len(fake_import) == 1  # imported once
    assert screens["clicks"] == [(OLD["screen_x"], OLD["screen_y"]), (320, 779 + 34)]
    assert "Replaced the previous version (iterate_me)" in msg


def test_reimport_keeps_item_with_other_body_count(screens):
    other = SELECTED[:-2] + [row("7 bodies", 328, 270), row("Delete", 779, 270)]
    screens["screens"] = [SIDEBAR + [OLD], other, SIDEBAR + [OLD], SIDEBAR + [OLD, NEW]]
    msg = server.open_in_shapr3d("iterate_me.step", versioned=False)
    assert len(screens["clicks"]) == 1 and screens["keys"] == ["escape"]
    assert "7 bodies" in msg and "left in place" in msg


def test_reimport_ambiguous_names_left_alone(screens):
    two = SIDEBAR + [row("› O iterate_m...", 147), row("› O iterate_me", 171)]
    screens["screens"] = [two, two, two + [row("› O iterate_me", 195)]]
    msg = server.open_in_shapr3d("iterate_me.step", versioned=False)
    assert screens["clicks"] == [] and "left them all in place" in msg


def test_new_model_is_added_not_replacing(screens):
    screens["screens"] = [SIDEBAR + [row("› O bracket", 147)]] * 2 + [
        SIDEBAR + [row("› O bracket", 147), row("› O iterate_me", 171)]]
    msg = server.open_in_shapr3d("iterate_me.step", versioned=False)
    assert screens["clicks"] == [] and "Replaced" not in msg


def test_failed_import_restores_deleted_version(screens, fake_import):
    screens["screens"] = [SIDEBAR + [OLD], SELECTED, SIDEBAR]  # nothing arrives
    with pytest.raises(RuntimeError, match="restored"):
        server.open_in_shapr3d("iterate_me.step", versioned=False)
    assert screens["keys"] == ["cmd+z"]
    assert len(fake_import) == 1  # never re-imported: a slow import would duplicate


def test_replace_false_never_touches_items(screens):
    screens["screens"] = [SIDEBAR + [OLD], SIDEBAR + [OLD, NEW]]
    server.open_in_shapr3d("iterate_me.step", replace=False, versioned=False)
    assert screens["clicks"] == []


def test_versioned_import_names_copy_and_deletes_previous_by_name(screens, fake_import):
    # manifest says v1 was imported last: the v1 row is deleted by name alone
    # (no body-count check), and the new import arrives as iterate_me_v2
    server.IMPORTS_MANIFEST.write_text('{"iterate_me": {"bodies": 99, "version": 1}}')
    v1, v2 = row("› O iterate_me_v1", 147), row("› O iterate_me_v2", 147)
    sel = SIDEBAR + [v1, row("Delete", 779, 270)]
    screens["screens"] = [SIDEBAR + [v1], sel, SIDEBAR, SIDEBAR, SIDEBAR + [v2]]
    msg = server.open_in_shapr3d("iterate_me.step")
    assert "iterate_me_v2.step" in fake_import[0]
    assert screens["clicks"] == [(v1["screen_x"], v1["screen_y"]), (320, 779 + 34)]
    assert "Replaced the previous version (iterate_me_v1)" in msg
    assert '"version": 2' in server.IMPORTS_MANIFEST.read_text()


def test_first_versioned_import_checks_legacy_item_body_count(screens, fake_import):
    # no version recorded yet: an unversioned item of the same name is only
    # replaced when its body count matches, as before
    other = SELECTED[:-2] + [row("7 bodies", 328, 270), row("Delete", 779, 270)]
    v1 = row("› O iterate_me_v1", 147)
    screens["screens"] = [SIDEBAR + [OLD], other, SIDEBAR + [OLD], SIDEBAR + [OLD, v1]]
    msg = server.open_in_shapr3d("iterate_me.step")
    assert "iterate_me_v1.step" in fake_import[0]
    assert "left in place" in msg


def test_body_count_tolerance():
    assert macos.body_count_close(112, {113})      # Shapr3D merged a vendor sliver
    assert not macos.body_count_close(120, {113})  # user added parts
    assert not macos.body_count_close(None, {113})
