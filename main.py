"""
zot extension: Kagi search and URL extraction tools.
Reads KAGI_API_KEY from the environment.

User-approval gate
------------------
Every tool call is held until the user explicitly approves or denies it
via an interactive panel.  The tool thread emits a submit_slash frame to
trigger /kagi-approve, which opens the panel via command_response.  The
panel shows Yes / No and is navigated with the keyboard.

Only one Kagi call can be pending at a time.  If the agent fires a
second call while one is already waiting, it is immediately denied.

Cache
-----
Search queries and extract URLs are stored in a local SQLite database.
When a new tool call arrives, the cache is checked first:
  - kagi_search: cosine-similarity match against stored queries
  - kagi_extract: exact URL-set match

If a match is found above the threshold, a "cache hit" panel is shown
that lets the user return the stored result (free) or proceed with a
live API call (costs money).
"""
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import openapi_client  # type: ignore

import cache as kagi_cache

# ---------------------------------------------------------------------------
# Wire-protocol helpers
# ---------------------------------------------------------------------------

_emit_lock = threading.Lock()

APPROVE_PANEL_ID = "kagi-approve"
CACHE_PANEL_ID   = "kagi-cache-hit"

def emit(obj: dict) -> None:
    with _emit_lock:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

def log(msg: str) -> None:
    sys.stderr.write(f"[kagi] {msg}\n")
    sys.stderr.flush()

def tool_error(call_id: str, msg: str) -> None:
    emit({
        "type": "tool_result",
        "id": call_id,
        "is_error": True,
        "content": [{"type": "text", "text": msg}],
    })

def tool_ok(call_id: str, text: str) -> None:
    emit({
        "type": "tool_result",
        "id": call_id,
        "content": [{"type": "text", "text": text}],
    })

def notify(level: str, message: str) -> None:
    emit({"type": "notify", "level": level, "message": message})

# ---------------------------------------------------------------------------
# Cache-hit approval panel
# ---------------------------------------------------------------------------
# Options: 0 = "Use cached result", 1 = "Make live API call"

_CACHE_OPTIONS = ["Use cached result  (free)", "Make live API call  (costs money)", "Deny entirely"]

class _CacheChoice:
    USE_CACHE = "use_cache"
    LIVE_CALL = "live_call"
    DENY      = "deny"


@dataclass
class _PendingCacheDecision:
    hit: kagi_cache.CacheHit
    event: threading.Event = field(default_factory=threading.Event)
    choice: str = _CacheChoice.USE_CACHE
    cursor: int = 0
    panel_open: bool = False


_cache_decision_lock = threading.Lock()
_cache_decision: Optional[_PendingCacheDecision] = None


def _render_cache_lines(p: _PendingCacheDecision) -> list[str]:
    import datetime
    when = datetime.datetime.fromtimestamp(p.hit.created_at).strftime("%Y-%m-%d %H:%M")
    sim_pct = int(p.hit.similarity * 100)
    lines = [
        "A cached result was found for this query.",
        f"  Original: {p.hit.original_query[:80]}",
        f"  Cached:   {when}  ·  similarity {sim_pct}%",
        "",
    ]
    for i, option in enumerate(_CACHE_OPTIONS):
        marker = ">" if i == p.cursor else " "
        lines.append(f"  {marker} {option}")
    lines.append("")
    lines.append("  esc / ctrl-c  →  deny entirely")
    return lines


def _push_cache_render(p: _PendingCacheDecision) -> None:
    emit({
        "type": "panel_render",
        "panel_id": CACHE_PANEL_ID,
        "title": "Kagi Cache Hit",
        "lines": _render_cache_lines(p),
        "footer": "↑/↓  move  ·  enter  confirm  ·  esc/ctrl-c  deny entirely",
    })


def _resolve_cache(p: _PendingCacheDecision, choice: str) -> None:
    p.panel_open = False
    p.choice = choice
    emit({"type": "panel_close", "panel_id": CACHE_PANEL_ID})
    emit({"type": "clear_notes"})
    p.event.set()


def _request_cache_decision(hit: kagi_cache.CacheHit) -> str:
    """
    Open the cache-hit panel and block until the user chooses.
    Returns a _CacheChoice constant: USE_CACHE, LIVE_CALL, or DENY.
    """
    global _cache_decision

    decision = _PendingCacheDecision(hit=hit)

    with _cache_decision_lock:
        if _cache_decision is not None:
            return _CacheChoice.USE_CACHE   # safe default: use cache
        _cache_decision = decision

    emit({"type": "submit_slash", "text": "/kagi-cache-hit"})
    decision.event.wait()

    with _cache_decision_lock:
        _cache_decision = None

    return decision.choice


def handle_cache_hit_command(cmd_id: str) -> None:
    with _cache_decision_lock:
        p = _cache_decision

    if p is None:
        emit({
            "type": "command_response",
            "id": cmd_id,
            "action": "display",
            "display": "No cache decision pending.",
        })
        return

    p.panel_open = True
    emit({
        "type": "command_response",
        "id": cmd_id,
        "action": "open_panel",
        "open_panel": {
            "id": CACHE_PANEL_ID,
            "title": "Kagi Cache Hit",
            "lines": _render_cache_lines(p),
            "footer": "↑/↓  move  ·  enter  confirm  ·  esc/ctrl-c  deny entirely",
        },
    })


def handle_cache_panel_key(frame: dict) -> None:
    if frame.get("panel_id") != CACHE_PANEL_ID:
        return

    with _cache_decision_lock:
        p = _cache_decision

    if p is None or not p.panel_open:
        return

    key = frame.get("key", "")

    if key == "up":
        p.cursor = max(0, p.cursor - 1)
        _push_cache_render(p)
    elif key in ("down", "tab"):
        p.cursor = min(len(_CACHE_OPTIONS) - 1, p.cursor + 1)
        _push_cache_render(p)
    elif key == "enter":
        choices = [_CacheChoice.USE_CACHE, _CacheChoice.LIVE_CALL, _CacheChoice.DENY]
        _resolve_cache(p, choices[p.cursor])
    elif key == "esc":
        _resolve_cache(p, _CacheChoice.DENY)
    elif key == "rune":
        text = frame.get("text", "")
        if text in ("\n", "\r"):
            choices = [_CacheChoice.USE_CACHE, _CacheChoice.LIVE_CALL, _CacheChoice.DENY]
            _resolve_cache(p, choices[p.cursor])


def handle_cache_panel_close(frame: dict) -> None:
    """Host closed the panel (e.g. ctrl-c in TUI) — treat as deny entirely."""
    if frame.get("panel_id") != CACHE_PANEL_ID:
        return
    with _cache_decision_lock:
        p = _cache_decision
    if p is None:
        return
    emit({"type": "clear_notes"})
    p.panel_open = False
    p.choice = _CacheChoice.DENY
    p.event.set()


# ---------------------------------------------------------------------------
# Approval gate (panel-based)
# ---------------------------------------------------------------------------

_OPTIONS = ["Yes", "No"]

@dataclass
class _PendingApproval:
    description: str
    event: threading.Event = field(default_factory=threading.Event)
    approved: bool = False
    cursor: int = 0
    panel_open: bool = False


_pending_lock = threading.Lock()
_pending: Optional[_PendingApproval] = None


def _render_lines(p: _PendingApproval) -> list[str]:
    lines = [
        "Allow this Kagi API call?",
        f"  {p.description}",
        "",
    ]
    for i, option in enumerate(_OPTIONS):
        marker = ">" if i == p.cursor else " "
        lines.append(f"  {marker} {option}")
    return lines


def _push_render(p: _PendingApproval) -> None:
    emit({
        "type": "panel_render",
        "panel_id": APPROVE_PANEL_ID,
        "title": "Kagi API Approval",
        "lines": _render_lines(p),
        "footer": "↑/↓ or tab to select  ·  enter to confirm  ·  esc to deny",
    })


def _resolve(p: _PendingApproval, approved: bool) -> None:
    p.panel_open = False
    p.approved = approved
    emit({"type": "panel_close", "panel_id": APPROVE_PANEL_ID})
    emit({"type": "clear_notes"})
    p.event.set()


def _request_approval(description: str) -> bool:
    global _pending

    approval = _PendingApproval(description=description)

    with _pending_lock:
        if _pending is not None:
            return False
        _pending = approval

    notify(
        "warn",
        f"Kagi API call pending (costs money): {description}",
    )
    emit({"type": "submit_slash", "text": "/kagi-approve"})

    approval.event.wait()

    with _pending_lock:
        _pending = None

    return approval.approved


def handle_panel_key(frame: dict) -> None:
    if frame.get("panel_id") != APPROVE_PANEL_ID:
        return

    with _pending_lock:
        p = _pending

    if p is None or not p.panel_open:
        return

    key = frame.get("key", "")

    if key == "up":
        p.cursor = max(0, p.cursor - 1)
        _push_render(p)
    elif key in ("down", "tab"):
        p.cursor = min(len(_OPTIONS) - 1, p.cursor + 1)
        _push_render(p)
    elif key == "enter":
        _resolve(p, approved=(p.cursor == 0))
    elif key == "esc":
        _resolve(p, approved=False)
    elif key == "rune":
        text = frame.get("text", "")
        if text in ("\n", "\r"):
            _resolve(p, approved=(p.cursor == 0))


def handle_panel_close(frame: dict) -> None:
    if frame.get("panel_id") != APPROVE_PANEL_ID:
        return

    with _pending_lock:
        p = _pending

    if p is None:
        return

    emit({"type": "clear_notes"})
    p.panel_open = False
    p.approved = False
    p.event.set()


# ---------------------------------------------------------------------------
# Kagi API client
# ---------------------------------------------------------------------------

def _client() -> openapi_client.ApiClient:
    api_key = os.environ.get("KAGI_API_KEY", "")
    if not api_key:
        raise RuntimeError("KAGI_API_KEY environment variable is not set")
    cfg = openapi_client.Configuration(access_token=api_key)
    return openapi_client.ApiClient(cfg)

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def handle_kagi_search(call_id: str, args: dict) -> None:
    query: str = args.get("query", "").strip()
    if not query:
        tool_error(call_id, "Missing required argument: query")
        return

    limit: int = int(args.get("limit", 10))
    limit = max(1, min(limit, 20))

    # --- Cache lookup ---
    hit = kagi_cache.search_cache_lookup(query)
    if hit is not None:
        log(f"cache hit for search query={query!r} sim={hit.similarity:.2f}")
        choice = _request_cache_decision(hit)
        if choice == _CacheChoice.USE_CACHE:
            tool_ok(call_id, hit.result)
            return
        if choice == _CacheChoice.DENY:
            tool_error(call_id, "Kagi search was denied by the user. Do not retry without asking the user first.")
            return
        # LIVE_CALL — fall through to approval + API

    # --- Approval gate ---
    if not _request_approval(f"kagi_search(query={query!r}, limit={limit})"):
        tool_error(call_id, "Kagi search was denied by the user. Do not retry without asking the user first.")
        return

    try:
        with _client() as api_client_instance:
            api = openapi_client.SearchApi(api_client_instance)
            request = openapi_client.SearchRequest(query=query, limit=limit)
            response = api.search(request)
    except RuntimeError as exc:
        tool_error(call_id, str(exc))
        return
    except openapi_client.ApiException as exc:
        tool_error(call_id, f"Kagi API error {exc.status}: {exc.reason}")
        return
    except Exception as exc:  # noqa: BLE001
        tool_error(call_id, f"Unexpected error: {exc}")
        return

    data = getattr(response, "data", None)
    results = getattr(data, "search", None) or []
    if not results:
        tool_ok(call_id, "No results found.")
        return

    lines: list[str] = [f"Search results for: {query}\n"]
    for i, item in enumerate(results, 1):
        title = getattr(item, "title", "") or ""
        url = getattr(item, "url", "") or ""
        snippet = getattr(item, "snippet", "") or ""
        lines.append(f"{i}. **{title}**")
        lines.append(f"   {url}")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")

    result_text = "\n".join(lines).strip()

    # --- Store in cache ---
    try:
        kagi_cache.search_cache_store(query, result_text)
        log(f"cached search result for query={query!r}")
    except Exception as exc:  # noqa: BLE001
        log(f"cache write failed: {exc}")

    tool_ok(call_id, result_text)


def handle_kagi_extract(call_id: str, args: dict) -> None:
    urls: list[str] = args.get("urls", [])
    if not urls:
        tool_error(call_id, "Missing required argument: urls (must be a non-empty list)")
        return
    if len(urls) > 10:
        tool_error(call_id, "Too many URLs: maximum is 10 per request")
        return

    # --- Cache lookup (exact URL set) ---
    hit = kagi_cache.extract_cache_lookup(urls)
    if hit is not None:
        log(f"cache hit for extract urls={urls}")
        choice = _request_cache_decision(hit)
        if choice == _CacheChoice.USE_CACHE:
            tool_ok(call_id, hit.result)
            return
        if choice == _CacheChoice.DENY:
            tool_error(call_id, "Kagi extract was denied by the user. Do not retry without asking the user first.")
            return
        # LIVE_CALL — fall through to approval + API

    # --- Approval gate ---
    short_urls = ", ".join(urls[:3]) + ("…" if len(urls) > 3 else "")
    if not _request_approval(f"kagi_extract(urls=[{short_urls}], count={len(urls)})"):
        tool_error(call_id, "Kagi extract was denied by the user. Do not retry without asking the user first.")
        return

    try:
        pages = [openapi_client.PageInput(url=u) for u in urls]
        with _client() as api_client_instance:
            api = openapi_client.ExtractApi(api_client_instance)
            request = openapi_client.ExtractRequest(pages=pages)
            response = api.extract_content(request)
    except RuntimeError as exc:
        tool_error(call_id, str(exc))
        return
    except openapi_client.ApiException as exc:
        tool_error(call_id, f"Kagi API error {exc.status}: {exc.reason}")
        return
    except Exception as exc:  # noqa: BLE001
        tool_error(call_id, f"Unexpected error: {exc}")
        return

    extracted = getattr(response, "data", None) or []
    if not extracted:
        tool_ok(call_id, "No content extracted.")
        return

    parts: list[str] = []
    for page in extracted:
        url = getattr(page, "url", "") or ""
        content = getattr(page, "markdown", "") or ""
        error = getattr(page, "error", None)
        parts.append(f"## {url}\n")
        if error:
            parts.append(f"*Error: {error}*\n")
        elif content:
            parts.append(content)
        else:
            parts.append("*(no content returned)*")
        parts.append("\n---\n")

    result_text = "\n".join(parts).strip()

    # --- Store in cache ---
    try:
        kagi_cache.extract_cache_store(urls, result_text)
        log(f"cached extract result for {len(urls)} url(s)")
    except Exception as exc:  # noqa: BLE001
        log(f"cache write failed: {exc}")

    tool_ok(call_id, result_text)

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

TOOL_HANDLERS: dict[str, Callable[[str, dict], None]] = {
    "kagi_search": handle_kagi_search,
    "kagi_extract": handle_kagi_extract,
}

# ---------------------------------------------------------------------------
# /kagi-approve command handler
# ---------------------------------------------------------------------------

def handle_approve_command(cmd_id: str) -> None:
    with _pending_lock:
        p = _pending

    if p is None:
        emit({
            "type": "command_response",
            "id": cmd_id,
            "action": "display",
            "display": "No Kagi API call is pending.",
        })
        return

    p.panel_open = True
    emit({
        "type": "command_response",
        "id": cmd_id,
        "action": "open_panel",
        "open_panel": {
            "id": APPROVE_PANEL_ID,
            "title": "Kagi API Approval",
            "lines": _render_lines(p),
            "footer": "↑/↓ or tab to select  ·  enter to confirm  ·  esc to deny",
        },
    })

# ---------------------------------------------------------------------------
# /kagi-cache command handler
# ---------------------------------------------------------------------------

def handle_cache_command(cmd_id: str, args: str) -> None:
    """Display cache statistics and recent entries."""
    try:
        stats = kagi_cache.cache_stats()
        recent = kagi_cache.list_recent(limit=10)
    except Exception as exc:  # noqa: BLE001
        emit({
            "type": "command_response",
            "id": cmd_id,
            "action": "display",
            "display": f"Cache error: {exc}",
        })
        return

    def fmt_bytes(b: int) -> str:
        if b < 1024:
            return f"{b} B"
        if b < 1024 * 1024:
            return f"{b/1024:.1f} KB"
        return f"{b/1024/1024:.1f} MB"

    lines = [
        "**Kagi Cache Statistics**",
        "",
        f"Searches: {stats['search_count']} entries  ({fmt_bytes(stats['search_bytes'])} compressed)",
        f"Extracts: {stats['extract_count']} entries  ({fmt_bytes(stats['extract_bytes'])} compressed)",
        "",
        "**Recent searches:**",
    ]

    import datetime
    for row in recent["searches"]:
        when = datetime.datetime.fromtimestamp(row["created_at"]).strftime("%Y-%m-%d %H:%M")
        lines.append(f"  [{when}] {row['query'][:70]}  ({fmt_bytes(row['compressed_size'])})")

    if not recent["searches"]:
        lines.append("  (none)")

    lines += ["", "**Recent extracts:**"]
    for row in recent["extracts"]:
        when = datetime.datetime.fromtimestamp(row["created_at"]).strftime("%Y-%m-%d %H:%M")
        urls = json.loads(row["urls_json"])
        label = urls[0][:60] + ("…" if len(urls) > 1 or len(urls[0]) > 60 else "")
        lines.append(f"  [{when}] {label}  ({fmt_bytes(row['compressed_size'])})")

    if not recent["extracts"]:
        lines.append("  (none)")

    emit({
        "type": "command_response",
        "id": cmd_id,
        "action": "display",
        "display": "\n".join(lines),
    })

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    emit({
        "type": "hello",
        "name": "kagi",
        "version": "1.1.0",
        "capabilities": ["tools", "commands", "panels"],
    })

    emit({
        "type": "register_command",
        "name": "kagi-approve",
        "description": "(internal) open the Kagi API approval panel",
    })

    emit({
        "type": "register_command",
        "name": "kagi-cache-hit",
        "description": "(internal) open the Kagi cache-hit decision panel",
    })

    emit({
        "type": "register_command",
        "name": "kagi-cache",
        "description": "Show Kagi cache statistics and recent cached queries",
    })

    emit({
        "type": "register_tool",
        "name": "kagi_search",
        "description": "Search the web using Kagi and return ranked results with titles, URLs, and snippets.",
        "schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 10, max 20)",
                },
            },
            "required": ["query"],
        },
    })

    emit({
        "type": "register_tool",
        "name": "kagi_extract",
        "description": "Extract the full markdown content of one or more URLs using Kagi.",
        "schema": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of HTTPS URLs to extract content from (max 10)",
                },
            },
            "required": ["urls"],
        },
    })

    emit({"type": "ready"})
    log("ready — waiting for tool calls")

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            msg = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            log(f"failed to parse frame: {exc}")
            continue

        msg_type = msg.get("type")

        if msg_type == "command_invoked":
            name = msg.get("name", "")
            cmd_id = msg.get("id", "")
            args_str = msg.get("args", "")
            if name == "kagi-approve":
                handle_approve_command(cmd_id)
            elif name == "kagi-cache-hit":
                handle_cache_hit_command(cmd_id)
            elif name == "kagi-cache":
                handle_cache_command(cmd_id, args_str)
            else:
                emit({"type": "command_response", "id": cmd_id, "action": "display",
                      "display": f"Unknown command: {name}"})

        elif msg_type == "tool_call":
            call_id = msg.get("id", "")
            name = msg.get("name", "")
            call_args = msg.get("args", {})
            log(f"tool_call: {name}  id={call_id}")
            handler = TOOL_HANDLERS.get(name)
            if handler:
                threading.Thread(
                    target=handler,
                    args=(call_id, call_args),
                    daemon=True,
                ).start()
            else:
                tool_error(call_id, f"Unknown tool: {name}")

        elif msg_type == "panel_key":
            # Route to the right panel handler by panel_id
            panel_id = msg.get("panel_id", "")
            if panel_id == CACHE_PANEL_ID:
                handle_cache_panel_key(msg)
            else:
                handle_panel_key(msg)

        elif msg_type == "panel_close":
            panel_id = msg.get("panel_id", "")
            if panel_id == CACHE_PANEL_ID:
                handle_cache_panel_close(msg)
            else:
                handle_panel_close(msg)

        elif msg_type == "shutdown":
            log("shutting down")
            emit({"type": "shutdown_ack"})
            break

        else:
            log(f"ignoring frame type={msg_type!r}")


if __name__ == "__main__":
    main()
