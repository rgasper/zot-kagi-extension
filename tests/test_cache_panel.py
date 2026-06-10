"""
Tests for the cache-hit decision panel in main.py.

Covers the three-way choice (USE_CACHE / LIVE_CALL / DENY),
esc/ctrl-c → deny, host-close → deny, cursor navigation,
and that the tool handlers honour each choice correctly.
"""
import threading
import time
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

import main
from main import (
    CACHE_PANEL_ID,
    _CacheChoice,
    _PendingCacheDecision,
    _cache_decision_lock,
    _render_cache_lines,
    _push_cache_render,
    handle_cache_panel_key,
    handle_cache_panel_close,
    handle_cache_hit_command,
)
import cache as kagi_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hit(query: str = "test query") -> kagi_cache.CacheHit:
    return kagi_cache.CacheHit(
        row_id=1,
        original_query=query,
        result="cached result text",
        created_at=time.time(),
        distance=0.2,
    )


def _run_cache_decision_in_thread(hit: kagi_cache.CacheHit) -> dict:
    """
    Run _request_cache_decision in a background thread, simulate the host
    dispatching /kagi-cache-hit back so the panel opens.

    Returns dict with keys:
      'choice'       — _CacheChoice constant once resolved
      'thread'       — Thread object
      'emitted'      — list of emitted JSON frames
      '_emit_lock'   — lock guarding emitted
    """
    result = {}
    emitted = []
    emit_lock = threading.Lock()
    slash_seen = threading.Event()

    def _capture_emit(obj):
        with emit_lock:
            emitted.append(obj)
        if obj.get("type") == "submit_slash" and obj.get("text") == "/kagi-cache-hit":
            slash_seen.set()

    def _worker():
        with patch("main.emit", side_effect=_capture_emit):
            result["choice"] = main._request_cache_decision(hit)

    def _host():
        if slash_seen.wait(timeout=2.0):
            with patch("main.emit", side_effect=_capture_emit):
                handle_cache_hit_command("cmd-test")

    t = threading.Thread(target=_worker, daemon=True)
    h = threading.Thread(target=_host, daemon=True)
    result["thread"] = t
    result["emitted"] = emitted
    result["_emit_lock"] = emit_lock
    t.start()
    h.start()
    return result


def _wait_for_panel_open(result: dict, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    lock = result["_emit_lock"]
    while time.monotonic() < deadline:
        with lock:
            snapshot = list(result["emitted"])
        if any(o.get("action") == "open_panel" for o in snapshot):
            return True
        time.sleep(0.01)
    return False


def _key(key: str, text: str = "") -> dict:
    f = {"type": "panel_key", "panel_id": CACHE_PANEL_ID, "key": key}
    if text:
        f["text"] = text
    return f


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

class TestCachePanelRender:
    def test_shows_original_query(self):
        p = _PendingCacheDecision(hit=_make_hit("best python frameworks"))
        lines = _render_cache_lines(p)
        assert any("best python frameworks" in l for l in lines)

    def test_shows_all_three_options(self):
        p = _PendingCacheDecision(hit=_make_hit())
        text = "\n".join(_render_cache_lines(p))
        assert "Use cached result" in text
        assert "Make live API call" in text
        assert "Deny entirely" in text

    def test_shows_esc_hint(self):
        p = _PendingCacheDecision(hit=_make_hit())
        text = "\n".join(_render_cache_lines(p))
        assert "esc" in text.lower() or "deny" in text.lower()

    def test_cursor_on_use_cache_by_default(self):
        p = _PendingCacheDecision(hit=_make_hit())
        lines = _render_cache_lines(p)
        use_cache_line = next(l for l in lines if "Use cached result" in l)
        assert use_cache_line.strip().startswith(">")

    def test_cursor_on_deny_when_cursor_2(self):
        p = _PendingCacheDecision(hit=_make_hit(), cursor=2)
        lines = _render_cache_lines(p)
        deny_line = next(l for l in lines if "Deny entirely" in l)
        assert deny_line.strip().startswith(">")

    def test_similarity_shown_as_percentage(self):
        p = _PendingCacheDecision(hit=_make_hit())
        text = "\n".join(_render_cache_lines(p))
        assert "%" in text


# ---------------------------------------------------------------------------
# Three-way choice via enter
# ---------------------------------------------------------------------------

class TestCachePanelChoices:
    def test_enter_on_use_cache_returns_use_cache(self):
        result = _run_cache_decision_in_thread(_make_hit())
        assert _wait_for_panel_open(result)
        # cursor=0 → USE_CACHE
        handle_cache_panel_key(_key("enter"))
        result["thread"].join(timeout=1.0)
        assert result["choice"] == _CacheChoice.USE_CACHE

    def test_enter_on_live_call_returns_live_call(self):
        result = _run_cache_decision_in_thread(_make_hit())
        assert _wait_for_panel_open(result)
        handle_cache_panel_key(_key("down"))   # cursor → 1
        handle_cache_panel_key(_key("enter"))
        result["thread"].join(timeout=1.0)
        assert result["choice"] == _CacheChoice.LIVE_CALL

    def test_enter_on_deny_returns_deny(self):
        result = _run_cache_decision_in_thread(_make_hit())
        assert _wait_for_panel_open(result)
        handle_cache_panel_key(_key("down"))   # → 1
        handle_cache_panel_key(_key("down"))   # → 2
        handle_cache_panel_key(_key("enter"))
        result["thread"].join(timeout=1.0)
        assert result["choice"] == _CacheChoice.DENY

    def test_rune_newline_on_live_call_returns_live_call(self):
        result = _run_cache_decision_in_thread(_make_hit())
        assert _wait_for_panel_open(result)
        handle_cache_panel_key(_key("down"))
        handle_cache_panel_key(_key("rune", text="\n"))
        result["thread"].join(timeout=1.0)
        assert result["choice"] == _CacheChoice.LIVE_CALL


# ---------------------------------------------------------------------------
# Esc / ctrl-c → deny entirely
# ---------------------------------------------------------------------------

class TestCachePanelEscDeny:
    def test_esc_returns_deny(self):
        result = _run_cache_decision_in_thread(_make_hit())
        assert _wait_for_panel_open(result)
        handle_cache_panel_key(_key("esc"))
        result["thread"].join(timeout=1.0)
        assert result["choice"] == _CacheChoice.DENY

    def test_esc_does_not_return_use_cache(self):
        result = _run_cache_decision_in_thread(_make_hit())
        assert _wait_for_panel_open(result)
        handle_cache_panel_key(_key("esc"))
        result["thread"].join(timeout=1.0)
        assert result["choice"] != _CacheChoice.USE_CACHE

    def test_esc_does_not_return_live_call(self):
        result = _run_cache_decision_in_thread(_make_hit())
        assert _wait_for_panel_open(result)
        handle_cache_panel_key(_key("esc"))
        result["thread"].join(timeout=1.0)
        assert result["choice"] != _CacheChoice.LIVE_CALL

    def test_host_close_returns_deny(self):
        result = _run_cache_decision_in_thread(_make_hit())
        assert _wait_for_panel_open(result)
        handle_cache_panel_close({"type": "panel_close", "panel_id": CACHE_PANEL_ID})
        result["thread"].join(timeout=1.0)
        assert result["choice"] == _CacheChoice.DENY

    def test_host_close_wrong_panel_id_ignored(self):
        result = _run_cache_decision_in_thread(_make_hit())
        assert _wait_for_panel_open(result)
        handle_cache_panel_close({"type": "panel_close", "panel_id": "other-panel"})
        assert result["thread"].is_alive()
        handle_cache_panel_key(_key("esc"))
        result["thread"].join(timeout=1.0)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

class TestCachePanelNavigation:
    def test_default_cursor_is_zero(self):
        result = _run_cache_decision_in_thread(_make_hit())
        assert _wait_for_panel_open(result)
        with _cache_decision_lock:
            p = main._cache_decision
        assert p.cursor == 0
        handle_cache_panel_key(_key("esc"))
        result["thread"].join(timeout=1.0)

    def test_down_increments_cursor(self):
        result = _run_cache_decision_in_thread(_make_hit())
        assert _wait_for_panel_open(result)
        handle_cache_panel_key(_key("down"))
        with _cache_decision_lock:
            p = main._cache_decision
        assert p.cursor == 1
        handle_cache_panel_key(_key("esc"))
        result["thread"].join(timeout=1.0)

    def test_tab_increments_cursor(self):
        result = _run_cache_decision_in_thread(_make_hit())
        assert _wait_for_panel_open(result)
        handle_cache_panel_key(_key("tab"))
        with _cache_decision_lock:
            p = main._cache_decision
        assert p.cursor == 1
        handle_cache_panel_key(_key("esc"))
        result["thread"].join(timeout=1.0)

    def test_down_clamps_at_deny(self):
        result = _run_cache_decision_in_thread(_make_hit())
        assert _wait_for_panel_open(result)
        for _ in range(10):
            handle_cache_panel_key(_key("down"))
        with _cache_decision_lock:
            p = main._cache_decision
        assert p.cursor == 2  # max index
        handle_cache_panel_key(_key("esc"))
        result["thread"].join(timeout=1.0)

    def test_up_clamps_at_zero(self):
        result = _run_cache_decision_in_thread(_make_hit())
        assert _wait_for_panel_open(result)
        handle_cache_panel_key(_key("up"))
        with _cache_decision_lock:
            p = main._cache_decision
        assert p.cursor == 0
        handle_cache_panel_key(_key("esc"))
        result["thread"].join(timeout=1.0)

    def test_down_up_returns_to_use_cache(self):
        result = _run_cache_decision_in_thread(_make_hit())
        assert _wait_for_panel_open(result)
        handle_cache_panel_key(_key("down"))
        handle_cache_panel_key(_key("up"))
        with _cache_decision_lock:
            p = main._cache_decision
        assert p.cursor == 0
        handle_cache_panel_key(_key("esc"))
        result["thread"].join(timeout=1.0)

    def test_wrong_panel_id_ignored(self):
        result = _run_cache_decision_in_thread(_make_hit())
        assert _wait_for_panel_open(result)
        handle_cache_panel_key({"type": "panel_key", "panel_id": "other", "key": "enter"})
        assert result["thread"].is_alive()
        handle_cache_panel_key(_key("esc"))
        result["thread"].join(timeout=1.0)


# ---------------------------------------------------------------------------
# Tool handler integration — deny from cache panel
# ---------------------------------------------------------------------------

def _make_search_hit():
    return kagi_cache.CacheHit(
        row_id=1,
        original_query="python frameworks",
        result="cached search result",
        created_at=time.time(),
        distance=0.1,
    )


def _make_extract_hit():
    return kagi_cache.CacheHit(
        row_id=1,
        original_query="https://example.com",
        result="cached page content",
        created_at=time.time(),
        distance=0.0,
    )


class TestToolHandlerCacheDeny:
    def test_search_deny_returns_error(self):
        buf = StringIO()
        with patch("main.kagi_cache.search_cache_lookup", return_value=_make_search_hit()):
            with patch("main._request_cache_decision", return_value=_CacheChoice.DENY):
                with patch("sys.stdout", buf):
                    main.handle_kagi_search("c1", {"query": "python frameworks"})
        import json
        frame = json.loads(buf.getvalue().strip())
        assert frame["is_error"] is True
        assert "denied" in frame["content"][0]["text"].lower()

    def test_search_deny_does_not_hit_api(self):
        api_called = []
        buf = StringIO()
        with patch("main.kagi_cache.search_cache_lookup", return_value=_make_search_hit()):
            with patch("main._request_cache_decision", return_value=_CacheChoice.DENY):
                with patch("main._request_approval", side_effect=lambda d: api_called.append(d) or True):
                    with patch("sys.stdout", buf):
                        main.handle_kagi_search("c1", {"query": "python frameworks"})
        assert not api_called

    def test_search_live_call_proceeds_to_approval(self):
        approval_called = []
        buf = StringIO()
        with patch("main.kagi_cache.search_cache_lookup", return_value=_make_search_hit()):
            with patch("main._request_cache_decision", return_value=_CacheChoice.LIVE_CALL):
                with patch("main._request_approval", side_effect=lambda d: approval_called.append(d) or False):
                    with patch("sys.stdout", buf):
                        main.handle_kagi_search("c1", {"query": "python frameworks"})
        assert approval_called

    def test_extract_deny_returns_error(self):
        buf = StringIO()
        with patch("main.kagi_cache.extract_cache_lookup", return_value=_make_extract_hit()):
            with patch("main._request_cache_decision", return_value=_CacheChoice.DENY):
                with patch("sys.stdout", buf):
                    main.handle_kagi_extract("c1", {"urls": ["https://example.com"]})
        import json
        frame = json.loads(buf.getvalue().strip())
        assert frame["is_error"] is True
        assert "denied" in frame["content"][0]["text"].lower()

    def test_extract_deny_does_not_hit_api(self):
        api_called = []
        buf = StringIO()
        with patch("main.kagi_cache.extract_cache_lookup", return_value=_make_extract_hit()):
            with patch("main._request_cache_decision", return_value=_CacheChoice.DENY):
                with patch("main._request_approval", side_effect=lambda d: api_called.append(d) or True):
                    with patch("sys.stdout", buf):
                        main.handle_kagi_extract("c1", {"urls": ["https://example.com"]})
        assert not api_called

    def test_extract_live_call_proceeds_to_approval(self):
        approval_called = []
        buf = StringIO()
        with patch("main.kagi_cache.extract_cache_lookup", return_value=_make_extract_hit()):
            with patch("main._request_cache_decision", return_value=_CacheChoice.LIVE_CALL):
                with patch("main._request_approval", side_effect=lambda d: approval_called.append(d) or False):
                    with patch("sys.stdout", buf):
                        main.handle_kagi_extract("c1", {"urls": ["https://example.com"]})
        assert approval_called
