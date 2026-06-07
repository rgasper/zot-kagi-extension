"""
zot extension: Kagi search and URL extraction tools.
Reads KAGI_API_KEY from the environment.
"""
import json
import os
import sys

import openapi_client

# ---------------------------------------------------------------------------
# Wire-protocol helpers
# ---------------------------------------------------------------------------

def emit(obj: dict) -> None:
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

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _client() -> openapi_client.ApiClient:
    api_key = os.environ.get("KAGI_API_KEY", "")
    if not api_key:
        raise RuntimeError("KAGI_API_KEY environment variable is not set")
    cfg = openapi_client.Configuration(access_token=api_key)
    return openapi_client.ApiClient(cfg)


def handle_kagi_search(call_id: str, args: dict) -> None:
    query: str = args.get("query", "").strip()
    if not query:
        tool_error(call_id, "Missing required argument: query")
        return

    limit: int = int(args.get("limit", 10))
    limit = max(1, min(limit, 20))

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
# Tool registry
# ---------------------------------------------------------------------------

TOOLS = {
    "kagi_search": handle_kagi_search,
    "kagi_extract": handle_kagi_extract,
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    # Handshake
    emit({
        "type": "hello",
        "name": "kagi",
        "version": "1.0.0",
        "capabilities": ["tools"],
    })

    # Register kagi_search
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

    # Register kagi_extract
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

        if msg_type == "tool_call":
            call_id = msg.get("id", "")
            name = msg.get("name", "")
            call_args = msg.get("args", {})
            log(f"tool_call: {name}  id={call_id}")
            handler = TOOLS.get(name)
            if handler:
                handler(call_id, call_args)
            else:
                tool_error(call_id, f"Unknown tool: {name}")

        elif msg_type == "shutdown":
            log("shutting down")
            emit({"type": "shutdown_ack"})
            break

        else:
            log(f"ignoring frame type={msg_type!r}")


if __name__ == "__main__":
    main()
