"""
zot extension: Kagi search and URL extraction tools.
Reads KAGI_API_KEY from the environment.

User-approval gate
------------------
Every tool call is held until the user explicitly approves or denies it.
The extension sends a `notify` describing the pending call, then blocks
the worker thread on a threading.Event.  The user types:
  /kagi-approve   → call proceeds, results returned to the agent
  /kagi-deny      → call is cancelled; agent receives an is_error result

Only one Kagi call can be pending at a time.  If the agent fires a second
call while one is already waiting, it is immediately denied.
"""
import json
import os
import sys
import threading
from dataclasses import dataclass, field
from typing import Callable

import openapi_client

# ---------------------------------------------------------------------------
# Wire-protocol helpers
# ---------------------------------------------------------------------------

_emit_lock = threading.Lock()

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
# Approval gate
# ---------------------------------------------------------------------------

@dataclass
class _PendingApproval:
    event: threading.Event = field(default_factory=threading.Event)
    approved: bool = False

_pending_lock = threading.Lock()
_pending: _PendingApproval | None = None


def _request_approval(description: str) -> bool:
    """
    Block the calling thread until the user runs /kagi-approve or /kagi-deny.
    Returns True if approved, False if denied or if another call is already pending.
    """
    global _pending

    approval = _PendingApproval()

    with _pending_lock:
        if _pending is not None:
            # Another call is already waiting — deny immediately
            return False
        _pending = approval

    notify(
        "warn",
        f"⏸ Kagi API call requested (costs money):\n"
        f"  {description}\n"
        f"  → /kagi-approve  to allow\n"
        f"  → /kagi-deny     to cancel",
    )

    approval.event.wait()  # blocks until /kagi-approve or /kagi-deny fires

    with _pending_lock:
        _pending = None

    return approval.approved


def handle_approve(cmd_id: str, _args: str) -> None:
    global _pending
    with _pending_lock:
        p = _pending
    if p is None:
        emit({"type": "command_response", "id": cmd_id, "action": "display",
              "display": "No Kagi call is pending."})
        return
    p.approved = True
    p.event.set()
    emit({"type": "command_response", "id": cmd_id, "action": "display",
          "display": "✅ Kagi call approved — proceeding."})


def handle_deny(cmd_id: str, _args: str) -> None:
    global _pending
    with _pending_lock:
        p = _pending
    if p is None:
        emit({"type": "command_response", "id": cmd_id, "action": "display",
              "display": "No Kagi call is pending."})
        return
    p.approved = False
    p.event.set()
    emit({"type": "command_response", "id": cmd_id, "action": "display",
          "display": "❌ Kagi call denied — the agent has been informed."})

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

    description = f"kagi_search(query={query!r}, limit={limit})"
    if not _request_approval(description):
        tool_error(
            call_id,
            "Kagi search was denied by the user. Do not retry without asking the user first.",
        )
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

    tool_ok(call_id, "\n".join(lines).strip())


def handle_kagi_extract(call_id: str, args: dict) -> None:
    urls: list[str] = args.get("urls", [])
    if not urls:
        tool_error(call_id, "Missing required argument: urls (must be a non-empty list)")
        return
    if len(urls) > 10:
        tool_error(call_id, "Too many URLs: maximum is 10 per request")
        return

    short_urls = ", ".join(urls[:3]) + ("…" if len(urls) > 3 else "")
    description = f"kagi_extract(urls=[{short_urls}], count={len(urls)})"
    if not _request_approval(description):
        tool_error(
            call_id,
            "Kagi extract was denied by the user. Do not retry without asking the user first.",
        )
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

    tool_ok(call_id, "\n".join(parts).strip())

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

TOOLS: dict[str, Callable[[str, dict], None]] = {
    "kagi_search": handle_kagi_search,
    "kagi_extract": handle_kagi_extract,
}

COMMANDS: dict[str, Callable[[str, str], None]] = {
    "kagi-approve": handle_approve,
    "kagi-deny": handle_deny,
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    emit({
        "type": "hello",
        "name": "kagi",
        "version": "1.0.0",
        "capabilities": ["tools", "commands"],
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

    emit({
        "type": "register_command",
        "name": "kagi-approve",
        "description": "Approve the pending Kagi API call",
    })

    emit({
        "type": "register_command",
        "name": "kagi-deny",
        "description": "Deny the pending Kagi API call",
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

        if msg_type == "tool_call":
            call_id = msg.get("id", "")
            name = msg.get("name", "")
            call_args = msg.get("args", {})
            log(f"tool_call: {name}  id={call_id}")
            handler = TOOLS.get(name)
            if handler:
                # Run on a worker thread so the approval wait doesn't
                # block the main I/O loop (which must keep reading stdin
                # to receive /kagi-approve or /kagi-deny).
                threading.Thread(
                    target=handler,
                    args=(call_id, call_args),
                    daemon=True,
                ).start()
            else:
                tool_error(call_id, f"Unknown tool: {name}")

        elif msg_type == "command_invoked":
            cmd_id = msg.get("id", "")
            name = msg.get("name", "")
            cmd_args = msg.get("args", "")
            log(f"command_invoked: {name}  id={cmd_id}")
            handler = COMMANDS.get(name)
            if handler:
                handler(cmd_id, cmd_args)
            else:
                emit({"type": "command_response", "id": cmd_id, "action": "display",
                      "display": f"Unknown command: {name}"})

        elif msg_type == "shutdown":
            # Unblock any waiting approval as a denied call so the
            # worker thread exits cleanly before we ack.
            with _pending_lock:
                p = _pending
            if p is not None:
                p.approved = False
                p.event.set()
            log("shutting down")
            emit({"type": "shutdown_ack"})
            break

        else:
            log(f"ignoring frame type={msg_type!r}")


if __name__ == "__main__":
    main()
