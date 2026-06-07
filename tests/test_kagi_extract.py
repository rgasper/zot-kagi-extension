"""
Tests for handle_kagi_extract in main.py.
"""
import json
from contextlib import contextmanager
from io import StringIO
from unittest.mock import MagicMock, patch

import openapi_client
from openapi_client.exceptions import (
    ApiException,
    UnauthorizedException,
    ForbiddenException,
    ServiceException,
    BadRequestException,
)

import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_page_output(url="https://example.com", markdown=None, error=None):
    return openapi_client.PageOutput(url=url, markdown=markdown, error=error)


def _make_extract_response(pages: list) -> openapi_client.ExtractResponse:
    meta = openapi_client.Meta(ms=42)
    return openapi_client.ExtractResponse(meta=meta, data=pages)


def _capture_extract(args: dict, env: dict | None = None) -> dict:
    env = {"KAGI_API_KEY": "test-key", **(env or {})}
    buf = StringIO()
    with patch.dict("os.environ", env, clear=False):
        with patch("sys.stdout", buf):
            main.handle_kagi_extract("call-1", args)
    return json.loads(buf.getvalue().strip())


@contextmanager
def _mock_extract_api(response=None, side_effect=None):
    mock_api = MagicMock()
    if side_effect is not None:
        mock_api.extract_content.side_effect = side_effect
    else:
        mock_api.extract_content.return_value = response

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("main._client", return_value=mock_client):
        with patch("openapi_client.ExtractApi", return_value=mock_api):
            yield mock_api


# ---------------------------------------------------------------------------
# Input-validation tests
# ---------------------------------------------------------------------------

class TestExtractInputValidation:
    def test_missing_urls_returns_error(self):
        frame = _capture_extract({})
        assert frame["is_error"] is True
        assert "urls" in frame["content"][0]["text"].lower()

    def test_empty_urls_list_returns_error(self):
        frame = _capture_extract({"urls": []})
        assert frame["is_error"] is True

    def test_too_many_urls_returns_error(self):
        urls = [f"https://example.com/{i}" for i in range(11)]
        frame = _capture_extract({"urls": urls})
        assert frame["is_error"] is True
        assert "10" in frame["content"][0]["text"]

    def test_exactly_10_urls_is_accepted(self):
        urls = [f"https://example.com/{i}" for i in range(10)]
        pages = [_make_page_output(url=u, markdown="content") for u in urls]
        resp = _make_extract_response(pages)
        with _mock_extract_api(response=resp):
            frame = _capture_extract({"urls": urls})
        assert frame.get("is_error") is not True

    def test_missing_api_key_returns_error(self):
        buf = StringIO()
        with patch.dict("os.environ", {}, clear=True):
            with patch("sys.stdout", buf):
                main.handle_kagi_extract("c1", {"urls": ["https://example.com"]})
        frame = json.loads(buf.getvalue().strip())
        assert frame["is_error"] is True
        assert "KAGI_API_KEY" in frame["content"][0]["text"]


# ---------------------------------------------------------------------------
# Successful response tests
# ---------------------------------------------------------------------------

class TestExtractSuccess:
    def test_single_url_markdown_returned(self):
        page = _make_page_output(
            url="https://example.com/article",
            markdown="# Hello\nThis is content.",
        )
        resp = _make_extract_response([page])
        with _mock_extract_api(response=resp):
            frame = _capture_extract({"urls": ["https://example.com/article"]})
        text = frame["content"][0]["text"]
        assert frame.get("is_error") is not True
        assert "https://example.com/article" in text
        assert "# Hello" in text
        assert "This is content." in text

    def test_multiple_urls_all_present(self):
        pages = [
            _make_page_output(url="https://a.com", markdown="content A"),
            _make_page_output(url="https://b.com", markdown="content B"),
        ]
        resp = _make_extract_response(pages)
        with _mock_extract_api(response=resp):
            frame = _capture_extract({"urls": ["https://a.com", "https://b.com"]})
        text = frame["content"][0]["text"]
        assert "https://a.com" in text
        assert "content A" in text
        assert "https://b.com" in text
        assert "content B" in text

    def test_page_with_error_field_shown(self):
        """PageOutput.error is populated when extraction fails for a URL."""
        page = _make_page_output(
            url="https://example.com/broken",
            error="fetch timeout",
        )
        resp = _make_extract_response([page])
        with _mock_extract_api(response=resp):
            frame = _capture_extract({"urls": ["https://example.com/broken"]})
        text = frame["content"][0]["text"]
        assert "fetch timeout" in text

    def test_page_with_no_markdown_and_no_error(self):
        """PageOutput with neither markdown nor error → no content returned message."""
        page = _make_page_output(url="https://example.com/empty", markdown=None, error=None)
        resp = _make_extract_response([page])
        with _mock_extract_api(response=resp):
            frame = _capture_extract({"urls": ["https://example.com/empty"]})
        text = frame["content"][0]["text"]
        # main.py reads page.content (wrong field); markdown is the real field.
        # Either way, no crash and the URL should appear.
        assert "https://example.com/empty" in text

    def test_empty_data_list_returns_no_content(self):
        resp = _make_extract_response([])
        with _mock_extract_api(response=resp):
            frame = _capture_extract({"urls": ["https://example.com"]})
        text = frame["content"][0]["text"]
        assert "No content" in text

    def test_none_data_returns_no_content(self):
        # ExtractResponse.data is non-nullable in the pydantic schema, so we
        # use a plain MagicMock to simulate a response whose .data is None.
        resp = MagicMock()
        resp.data = None
        with _mock_extract_api(response=resp):
            frame = _capture_extract({"urls": ["https://example.com"]})
        text = frame["content"][0]["text"]
        assert "No content" in text

    def test_pages_sent_to_sdk_match_urls(self):
        page = _make_page_output(url="https://example.com", markdown="x")
        resp = _make_extract_response([page])
        with _mock_extract_api(response=resp) as mock_api:
            _capture_extract({"urls": ["https://example.com", "https://other.com"]})
        req = mock_api.extract_content.call_args[0][0]
        sent_urls = [p.url for p in req.pages]
        assert sent_urls == ["https://example.com", "https://other.com"]

    def test_tool_result_id_preserved(self):
        page = _make_page_output(url="https://example.com", markdown="hi")
        resp = _make_extract_response([page])
        buf = StringIO()
        with patch.dict("os.environ", {"KAGI_API_KEY": "k"}):
            with patch("sys.stdout", buf):
                with _mock_extract_api(response=resp):
                    main.handle_kagi_extract("xyz-42", {"urls": ["https://example.com"]})
        frame = json.loads(buf.getvalue().strip())
        assert frame["id"] == "xyz-42"

    def test_markdown_field_used_not_content(self):
        """Regression: PageOutput.markdown is the correct field, not .content."""
        page = _make_page_output(url="https://example.com", markdown="the real content")
        resp = _make_extract_response([page])
        with _mock_extract_api(response=resp):
            frame = _capture_extract({"urls": ["https://example.com"]})
        text = frame["content"][0]["text"]
        # If main.py reads .content (wrong), this string won't appear
        assert "the real content" in text


# ---------------------------------------------------------------------------
# API error / exception tests
# ---------------------------------------------------------------------------

class TestExtractApiErrors:
    def test_401_unauthorized(self):
        exc = UnauthorizedException(status=401, reason="Unauthorized")
        with _mock_extract_api(side_effect=exc):
            frame = _capture_extract({"urls": ["https://example.com"]})
        assert frame["is_error"] is True
        assert "401" in frame["content"][0]["text"]

    def test_403_forbidden(self):
        exc = ForbiddenException(status=403, reason="Forbidden")
        with _mock_extract_api(side_effect=exc):
            frame = _capture_extract({"urls": ["https://example.com"]})
        assert frame["is_error"] is True
        assert "403" in frame["content"][0]["text"]

    def test_429_rate_limited(self):
        exc = ApiException(status=429, reason="Too Many Requests")
        with _mock_extract_api(side_effect=exc):
            frame = _capture_extract({"urls": ["https://example.com"]})
        assert frame["is_error"] is True
        assert "429" in frame["content"][0]["text"]

    def test_500_server_error(self):
        exc = ServiceException(status=500, reason="Internal Server Error")
        with _mock_extract_api(side_effect=exc):
            frame = _capture_extract({"urls": ["https://example.com"]})
        assert frame["is_error"] is True
        assert "500" in frame["content"][0]["text"]

    def test_400_bad_request(self):
        exc = BadRequestException(status=400, reason="Bad Request")
        with _mock_extract_api(side_effect=exc):
            frame = _capture_extract({"urls": ["https://example.com"]})
        assert frame["is_error"] is True
        assert "400" in frame["content"][0]["text"]

    def test_generic_exception_caught(self):
        with _mock_extract_api(side_effect=TimeoutError("timed out")):
            frame = _capture_extract({"urls": ["https://example.com"]})
        assert frame["is_error"] is True
        assert "timed out" in frame["content"][0]["text"]

    def test_error_frame_type_is_tool_result(self):
        exc = ApiException(status=503, reason="Service Unavailable")
        with _mock_extract_api(side_effect=exc):
            frame = _capture_extract({"urls": ["https://example.com"]})
        assert frame["type"] == "tool_result"
