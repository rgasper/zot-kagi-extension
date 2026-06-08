"""
Tests for the interactive panel approval gate in main.py.

Covers:
- _request_approval blocks until resolved
- handle_panel_key: up/down moves cursor, enter approves/denies, esc denies
- handle_panel_close: host-side close counts as deny
- concurrent call while one is pending → immediate deny
- shutdown unblocks a waiting approval as deny
"""
import json
import threading
import time
from io import StringIO
from unittest.mock import patch

import main
from main import (
    PANEL_ID,
    _PendingApproval,
    _pending_lock,
    handle_panel_key,
    handle_panel_close,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_approval_in_thread(description: str) -> dict:
    """
    Runs _request_approval in a background thread.
    Returns a dict with keys:
      'result'  – True/False once resolved
      'thread'  – the Thread object
      'emitted' – list of JSON objects written to stdout during approval
    """
    result = {}
    emitted = []

    def _capture_emit(obj):
        emitted.append(obj)

    def _worker():
        with patch("main.emit", side_effect=_capture_emit):
            result["value"] = main._request_approval(description)

    t = threading.Thread(target=_worker, daemon=True)
    result["thread"] = t
    result["emitted"] = emitted
    t.start()
    return result


def _wait_for_panel_open(result: dict, timeout: float = 1.0) -> bool:
    """Wait until the approval thread has opened the panel (emitted open_panel)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for obj in result["emitted"]:
            if obj.get("action") == "open_panel":
                return True
        time.sleep(0.01)
    return False


def _key(key: str, text: str = "") -> dict:
    frame = {"type": "panel_key", "panel_id": PANEL_ID, "key": key}
    if text:
        frame["text"] = text
    return frame


# ---------------------------------------------------------------------------
# Basic approval flow
# ---------------------------------------------------------------------------

class TestRequestApproval:
    def test_blocks_until_resolved(self):
        result = _run_approval_in_thread("test call")
        assert _wait_for_panel_open(result), "panel never opened"
        # Thread should still be alive (blocked)
        assert result["thread"].is_alive()
        # Resolve it
        handle_panel_key(_key("enter"))
        result["thread"].join(timeout=1.0)
        assert not result["thread"].is_alive()

    def test_opens_panel_with_description(self):
        result = _run_approval_in_thread("kagi_search(query='cats')")
        assert _wait_for_panel_open(result)
        open_panel_frames = [
            o for o in result["emitted"] if o.get("action") == "open_panel"
        ]
        assert open_panel_frames
        lines = open_panel_frames[0]["open_panel"]["lines"]
        assert any("kagi_search" in l for l in lines)
        # Clean up
        handle_panel_key(_key("esc"))
        result["thread"].join(timeout=1.0)

    def test_emits_notify_before_panel(self):
        result = _run_approval_in_thread("some call")
        assert _wait_for_panel_open(result)
        notify_frames = [o for o in result["emitted"] if o.get("type") == "notify"]
        assert notify_frames, "expected at least one notify frame"
        handle_panel_key(_key("esc"))
        result["thread"].join(timeout=1.0)

    def test_concurrent_call_denied_immediately(self):
        """A second _request_approval while one is pending should return False immediately."""
        result1 = _run_approval_in_thread("first call")
        assert _wait_for_panel_open(result1)

        # Second call — should return False without blocking
        result2 = {}
        emitted2 = []
        def _worker2():
            with patch("main.emit", side_effect=lambda o: emitted2.append(o)):
                result2["value"] = main._request_approval("second call")
        t2 = threading.Thread(target=_worker2, daemon=True)
        t2.start()
        t2.join(timeout=1.0)
        assert not t2.is_alive(), "second call should have returned immediately"
        assert result2["value"] is False

        # Clean up first
        handle_panel_key(_key("esc"))
        result1["thread"].join(timeout=1.0)

    def test_pending_cleared_after_resolve(self):
        result = _run_approval_in_thread("call")
        assert _wait_for_panel_open(result)
        handle_panel_key(_key("enter"))
        result["thread"].join(timeout=1.0)

        # _pending should be None now
        with _pending_lock:
            assert main._pending is None


# ---------------------------------------------------------------------------
# handle_panel_key — cursor navigation
# ---------------------------------------------------------------------------

class TestPanelKeyNavigation:
    def test_default_cursor_is_yes(self):
        result = _run_approval_in_thread("nav test")
        assert _wait_for_panel_open(result)
        with _pending_lock:
            p = main._pending
        assert p is not None
        assert p.cursor == 0  # 0 = Yes
        handle_panel_key(_key("esc"))
        result["thread"].join(timeout=1.0)

    def test_down_moves_to_no(self):
        result = _run_approval_in_thread("nav test")
        assert _wait_for_panel_open(result)
        handle_panel_key(_key("down"))
        with _pending_lock:
            p = main._pending
        assert p.cursor == 1
        handle_panel_key(_key("esc"))
        result["thread"].join(timeout=1.0)

    def test_tab_moves_to_no(self):
        result = _run_approval_in_thread("nav test")
        assert _wait_for_panel_open(result)
        handle_panel_key(_key("tab"))
        with _pending_lock:
            p = main._pending
        assert p.cursor == 1
        handle_panel_key(_key("esc"))
        result["thread"].join(timeout=1.0)

    def test_down_clamps_at_bottom(self):
        result = _run_approval_in_thread("nav test")
        assert _wait_for_panel_open(result)
        handle_panel_key(_key("down"))
        handle_panel_key(_key("down"))  # already at bottom
        with _pending_lock:
            p = main._pending
        assert p.cursor == 1
        handle_panel_key(_key("esc"))
        result["thread"].join(timeout=1.0)

    def test_up_clamps_at_top(self):
        result = _run_approval_in_thread("nav test")
        assert _wait_for_panel_open(result)
        handle_panel_key(_key("up"))  # already at top
        with _pending_lock:
            p = main._pending
        assert p.cursor == 0
        handle_panel_key(_key("esc"))
        result["thread"].join(timeout=1.0)

    def test_down_then_up_returns_to_yes(self):
        result = _run_approval_in_thread("nav test")
        assert _wait_for_panel_open(result)
        handle_panel_key(_key("down"))
        handle_panel_key(_key("up"))
        with _pending_lock:
            p = main._pending
        assert p.cursor == 0
        handle_panel_key(_key("esc"))
        result["thread"].join(timeout=1.0)


# ---------------------------------------------------------------------------
# handle_panel_key — confirm / cancel
# ---------------------------------------------------------------------------

class TestPanelKeyConfirm:
    def test_enter_on_yes_approves(self):
        result = _run_approval_in_thread("enter yes")
        assert _wait_for_panel_open(result)
        # cursor starts at Yes (0)
        handle_panel_key(_key("enter"))
        result["thread"].join(timeout=1.0)
        assert result["value"] is True

    def test_enter_on_no_denies(self):
        result = _run_approval_in_thread("enter no")
        assert _wait_for_panel_open(result)
        handle_panel_key(_key("down"))   # move to No
        handle_panel_key(_key("enter"))
        result["thread"].join(timeout=1.0)
        assert result["value"] is False

    def test_esc_denies(self):
        result = _run_approval_in_thread("esc test")
        assert _wait_for_panel_open(result)
        handle_panel_key(_key("esc"))
        result["thread"].join(timeout=1.0)
        assert result["value"] is False

    def test_rune_newline_on_yes_approves(self):
        result = _run_approval_in_thread("rune newline yes")
        assert _wait_for_panel_open(result)
        handle_panel_key(_key("rune", text="\n"))
        result["thread"].join(timeout=1.0)
        assert result["value"] is True

    def test_rune_newline_on_no_denies(self):
        result = _run_approval_in_thread("rune newline no")
        assert _wait_for_panel_open(result)
        handle_panel_key(_key("down"))
        handle_panel_key(_key("rune", text="\n"))
        result["thread"].join(timeout=1.0)
        assert result["value"] is False

    def test_rune_non_newline_ignored(self):
        result = _run_approval_in_thread("rune ignored")
        assert _wait_for_panel_open(result)
        handle_panel_key(_key("rune", text="x"))
        # thread should still be alive
        assert result["thread"].is_alive()
        handle_panel_key(_key("esc"))
        result["thread"].join(timeout=1.0)

    def test_enter_closes_panel(self):
        emitted = []
        result = _run_approval_in_thread("close on enter")
        assert _wait_for_panel_open(result)
        # Patch emit to capture post-confirm frames
        with patch("main.emit", side_effect=lambda o: emitted.append(o)):
            handle_panel_key(_key("enter"))
        result["thread"].join(timeout=1.0)
        close_frames = [o for o in emitted if o.get("type") == "panel_close"]
        assert close_frames

    def test_enter_emits_clear_notes(self):
        emitted = []
        result = _run_approval_in_thread("clear notes")
        assert _wait_for_panel_open(result)
        with patch("main.emit", side_effect=lambda o: emitted.append(o)):
            handle_panel_key(_key("enter"))
        result["thread"].join(timeout=1.0)
        clear_frames = [o for o in emitted if o.get("type") == "clear_notes"]
        assert clear_frames


# ---------------------------------------------------------------------------
# handle_panel_key — wrong panel_id ignored
# ---------------------------------------------------------------------------

class TestPanelKeyWrongId:
    def test_wrong_panel_id_ignored(self):
        result = _run_approval_in_thread("wrong id test")
        assert _wait_for_panel_open(result)
        # Send enter for a different panel — should not resolve
        handle_panel_key({"type": "panel_key", "panel_id": "other-panel", "key": "enter"})
        assert result["thread"].is_alive()
        # Clean up
        handle_panel_key(_key("esc"))
        result["thread"].join(timeout=1.0)


# ---------------------------------------------------------------------------
# handle_panel_close
# ---------------------------------------------------------------------------

class TestPanelClose:
    def test_host_close_denies(self):
        result = _run_approval_in_thread("host close")
        assert _wait_for_panel_open(result)
        handle_panel_close({"type": "panel_close", "panel_id": PANEL_ID})
        result["thread"].join(timeout=1.0)
        assert result["value"] is False

    def test_host_close_wrong_id_ignored(self):
        result = _run_approval_in_thread("host close wrong id")
        assert _wait_for_panel_open(result)
        handle_panel_close({"type": "panel_close", "panel_id": "other"})
        assert result["thread"].is_alive()
        handle_panel_key(_key("esc"))
        result["thread"].join(timeout=1.0)

    def test_host_close_emits_clear_notes(self):
        emitted = []
        result = _run_approval_in_thread("host close clear notes")
        assert _wait_for_panel_open(result)
        with patch("main.emit", side_effect=lambda o: emitted.append(o)):
            handle_panel_close({"type": "panel_close", "panel_id": PANEL_ID})
        result["thread"].join(timeout=1.0)
        clear_frames = [o for o in emitted if o.get("type") == "clear_notes"]
        assert clear_frames

    def test_host_close_when_no_pending_is_noop(self):
        # Should not raise
        handle_panel_close({"type": "panel_close", "panel_id": PANEL_ID})


# ---------------------------------------------------------------------------
# Render output
# ---------------------------------------------------------------------------

class TestPanelRender:
    def test_render_lines_contain_description(self):
        p = _PendingApproval(description="kagi_search(query='test')")
        lines = main._render_lines(p)
        assert any("kagi_search" in l for l in lines)

    def test_render_lines_show_yes_no(self):
        p = _PendingApproval(description="test")
        lines = main._render_lines(p)
        combined = "\n".join(lines)
        assert "Yes" in combined
        assert "No" in combined

    def test_render_lines_cursor_on_yes_by_default(self):
        p = _PendingApproval(description="test")
        lines = main._render_lines(p)
        yes_line = next(l for l in lines if "Yes" in l)
        assert yes_line.strip().startswith(">")

    def test_render_lines_cursor_on_no_when_cursor_1(self):
        p = _PendingApproval(description="test", cursor=1)
        lines = main._render_lines(p)
        no_line = next(l for l in lines if "No" in l)
        assert no_line.strip().startswith(">")

    def test_push_render_emits_panel_render(self):
        emitted = []
        p = _PendingApproval(description="test", panel_open=True)
        with patch("main.emit", side_effect=lambda o: emitted.append(o)):
            main._push_render(p)
        assert emitted[0]["type"] == "panel_render"
        assert emitted[0]["panel_id"] == PANEL_ID
