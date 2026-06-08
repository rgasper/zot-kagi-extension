#!/usr/bin/env python3
"""
Integration test: sends a real search request through handle_kagi_search
and a real extract request through handle_kagi_extract.

Usage:
    uv run python tests/integration_test.py <KAGI_API_KEY>
"""
import json
import os
import sys
from io import StringIO
from unittest.mock import patch

# Make sure we can import main from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main


def capture(fn, call_id, args, api_key):
    buf = StringIO()
    with patch.dict("os.environ", {"KAGI_API_KEY": api_key}, clear=False):
        with patch("sys.stdout", buf):
            with patch("main._request_approval", return_value=True):
                fn(call_id, args)
    return json.loads(buf.getvalue().strip())


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def run(api_key: str):
    # ------------------------------------------------------------------ #
    # 1. Search
    # ------------------------------------------------------------------ #
    section("kagi_search: 'what is the most popular cat breed'")
    frame = capture(
        main.handle_kagi_search,
        "integ-search-1",
        {"query": "what is the most popular cat breed", "limit": 5},
        api_key,
    )

    if frame.get("is_error"):
        print(f"ERROR: {frame['content'][0]['text']}")
        sys.exit(1)

    text = frame["content"][0]["text"]
    print(text)

    # Pull the first URL out of the result text so we can extract it
    first_url = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("https://"):
            first_url = line
            break

    # ------------------------------------------------------------------ #
    # 2. Extract (uses first URL from search if we got one)
    # ------------------------------------------------------------------ #
    extract_url = first_url or "https://en.wikipedia.org/wiki/Cat_breeds"
    section(f"kagi_extract: {extract_url}")
    frame2 = capture(
        main.handle_kagi_extract,
        "integ-extract-1",
        {"urls": [extract_url]},
        api_key,
    )

    if frame2.get("is_error"):
        print(f"ERROR: {frame2['content'][0]['text']}")
        sys.exit(1)

    content = frame2["content"][0]["text"]
    # Print just the first 1500 chars so the terminal doesn't flood
    print(content[:1500])
    if len(content) > 1500:
        print(f"\n... ({len(content) - 1500} more characters truncated)")

    section("All integration checks passed ✓")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: uv run python {sys.argv[0]} <KAGI_API_KEY>")
        sys.exit(1)
    run(sys.argv[1])
