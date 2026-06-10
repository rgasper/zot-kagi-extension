"""
Tests for cache.py — sqlite-vec backed cache with model2vec embeddings.

All DB tests use a fresh tmpdir path so nothing touches the real cache file.
The embedding model (potion-base-8M) is loaded once per session via a
module-level autouse fixture; individual tests reuse it without re-downloading.
"""
from __future__ import annotations

import json
import time
import zlib
from pathlib import Path

import numpy as np
import pytest

import cache as c


# ---------------------------------------------------------------------------
# Session-scoped model warm-up (avoids repeated HF downloads in CI)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def warm_model():
    """Pre-load the embedding model once for the whole test session."""
    c._get_model()


# ---------------------------------------------------------------------------
# Per-test fresh DB
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path) -> Path:
    return tmp_path / "test_kagi_cache.db"


# ---------------------------------------------------------------------------
# Compression round-trip
# ---------------------------------------------------------------------------

class TestCompression:
    def test_roundtrip_ascii(self):
        original = "Hello, world! This is a test."
        assert c.decompress(c.compress(original)) == original

    def test_roundtrip_unicode(self):
        original = "日本語テスト 🎉 émojis"
        assert c.decompress(c.compress(original)) == original

    def test_roundtrip_large(self):
        original = "a" * 100_000
        compressed = c.compress(original)
        assert len(compressed) < len(original)
        assert c.decompress(compressed) == original

    def test_compression_reduces_size_for_repetitive_text(self):
        original = "kagi search result\n" * 500
        assert len(c.compress(original)) < len(original) // 5

    def test_compress_returns_bytes(self):
        assert isinstance(c.compress("hi"), bytes)

    def test_decompress_returns_str(self):
        assert isinstance(c.decompress(c.compress("hi")), str)


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

class TestEmbed:
    def test_returns_numpy_array(self):
        v = c.embed("test query")
        assert isinstance(v, np.ndarray)

    def test_correct_dimension(self):
        v = c.embed("test query")
        assert v.shape == (c.EMBEDDING_DIM,)

    def test_dtype_float32(self):
        v = c.embed("test query")
        assert v.dtype == np.float32

    def test_different_texts_produce_different_vectors(self):
        a = c.embed("quantum physics")
        b = c.embed("chocolate cake recipe")
        assert not np.allclose(a, b)

    def test_same_text_produces_same_vector(self):
        a = c.embed("reproducible query")
        b = c.embed("reproducible query")
        assert np.allclose(a, b)

    def test_similar_queries_are_close(self):
        a = c.embed("best python web frameworks")
        b = c.embed("top python web frameworks")
        dist = float(np.linalg.norm(a - b))
        assert dist < c.DISTANCE_THRESHOLD

    def test_dissimilar_queries_are_far(self):
        a = c.embed("quantum physics equations")
        b = c.embed("chocolate cake recipe")
        dist = float(np.linalg.norm(a - b))
        assert dist > c.DISTANCE_THRESHOLD


# ---------------------------------------------------------------------------
# Search cache: store and lookup
# ---------------------------------------------------------------------------

class TestSearchCache:
    def test_exact_query_returned(self, db):
        c.search_cache_store("python async tutorial", "result text", db_path=db)
        hit = c.search_cache_lookup("python async tutorial", db_path=db)
        assert hit is not None
        assert hit.result == "result text"
        assert hit.distance < 0.01  # near-zero for exact same string

    def test_semantically_similar_query_matched(self, db):
        c.search_cache_store("best python web frameworks", "cached result", db_path=db)
        hit = c.search_cache_lookup("top python web frameworks", db_path=db)
        assert hit is not None
        assert hit.result == "cached result"

    def test_semantic_match_tensorflow(self, db):
        c.search_cache_store("how to install tensorflow", "tf result", db_path=db)
        hit = c.search_cache_lookup("tensorflow installation guide", db_path=db)
        assert hit is not None
        assert hit.result == "tf result"

    def test_dissimilar_query_not_matched(self, db):
        c.search_cache_store("quantum physics equations", "physics result", db_path=db)
        hit = c.search_cache_lookup("recipe for chocolate cake", db_path=db)
        assert hit is None

    def test_empty_cache_returns_none(self, db):
        assert c.search_cache_lookup("anything", db_path=db) is None

    def test_threshold_filters_borderline(self, db):
        c.search_cache_store("python tutorial beginners", "result", db_path=db)
        # exact match should always pass
        hit = c.search_cache_lookup("python tutorial beginners", db_path=db, threshold=0.01)
        assert hit is not None
        # completely different should fail at any threshold
        hit2 = c.search_cache_lookup("french cooking recipes pasta", db_path=db, threshold=c.DISTANCE_THRESHOLD)
        assert hit2 is None

    def test_nearest_of_multiple_returned(self, db):
        c.search_cache_store("python web framework flask", "flask result", db_path=db)
        c.search_cache_store("python web framework django", "django result", db_path=db)
        # Query clearly closer to django
        hit = c.search_cache_lookup("django python web framework", db_path=db)
        assert hit is not None
        assert hit.result == "django result"

    def test_created_at_is_recent(self, db):
        before = time.time()
        c.search_cache_store("test query", "result", db_path=db)
        after = time.time()
        hit = c.search_cache_lookup("test query", db_path=db)
        assert before <= hit.created_at <= after

    def test_original_query_preserved(self, db):
        c.search_cache_store("original exact query stored here", "result", db_path=db)
        hit = c.search_cache_lookup("original exact query stored here", db_path=db)
        assert hit.original_query == "original exact query stored here"

    def test_overwrite_exact_query_keeps_one_row(self, db):
        c.search_cache_store("duplicate query test", "first result", db_path=db)
        c.search_cache_store("duplicate query test", "second result", db_path=db)
        hit = c.search_cache_lookup("duplicate query test", db_path=db)
        assert hit.result == "second result"
        # Only one row in meta table
        import sqlite3, sqlite_vec as sv
        conn = sqlite3.connect(str(db))
        conn.enable_load_extension(True); sv.load(conn); conn.enable_load_extension(False)
        count = conn.execute(
            "SELECT COUNT(*) FROM search_meta WHERE query='duplicate query test'"
        ).fetchone()[0]
        conn.close()
        assert count == 1

    def test_result_is_compressed_on_disk(self, db):
        result_text = "This is my search result " * 100
        c.search_cache_store("compression test query", result_text, db_path=db)
        import sqlite3, sqlite_vec as sv
        conn = sqlite3.connect(str(db))
        conn.enable_load_extension(True); sv.load(conn); conn.enable_load_extension(False)
        row = conn.execute(
            "SELECT result_blob FROM search_meta WHERE query='compression test query'"
        ).fetchone()
        conn.close()
        blob = row[0]
        assert isinstance(blob, bytes)
        assert len(blob) < len(result_text)
        assert zlib.decompress(blob).decode() == result_text

    def test_similarity_property_near_one_for_exact(self, db):
        c.search_cache_store("similarity test query", "result", db_path=db)
        hit = c.search_cache_lookup("similarity test query", db_path=db)
        assert hit.similarity > 0.95

    def test_similarity_property_lower_for_paraphrase(self, db):
        c.search_cache_store("best python web frameworks", "result", db_path=db)
        hit = c.search_cache_lookup("top python web frameworks", db_path=db)
        assert hit is not None
        # Should be lower than exact but still positive
        assert 0.0 < hit.similarity < 1.0


# ---------------------------------------------------------------------------
# Extract cache: store and lookup  (exact URL-set match, no vectors)
# ---------------------------------------------------------------------------

class TestExtractCache:
    def test_exact_url_set_matched(self, db):
        urls = ["https://example.com/page1", "https://example.com/page2"]
        c.extract_cache_store(urls, "extracted content", db_path=db)
        hit = c.extract_cache_lookup(urls, db_path=db)
        assert hit is not None
        assert hit.result == "extracted content"
        assert hit.distance == 0.0

    def test_url_order_independent(self, db):
        urls = ["https://b.com", "https://a.com"]
        c.extract_cache_store(urls, "content", db_path=db)
        hit = c.extract_cache_lookup(["https://a.com", "https://b.com"], db_path=db)
        assert hit is not None

    def test_different_url_not_matched(self, db):
        c.extract_cache_store(["https://example.com/page1"], "content", db_path=db)
        hit = c.extract_cache_lookup(["https://example.com/page2"], db_path=db)
        assert hit is None

    def test_superset_not_matched(self, db):
        c.extract_cache_store(["https://a.com"], "content", db_path=db)
        hit = c.extract_cache_lookup(["https://a.com", "https://b.com"], db_path=db)
        assert hit is None

    def test_empty_cache_returns_none(self, db):
        assert c.extract_cache_lookup(["https://example.com"], db_path=db) is None

    def test_overwrite_same_url_set(self, db):
        urls = ["https://example.com"]
        c.extract_cache_store(urls, "old content", db_path=db)
        c.extract_cache_store(urls, "new content", db_path=db)
        hit = c.extract_cache_lookup(urls, db_path=db)
        assert hit.result == "new content"

    def test_result_compressed_on_disk(self, db):
        content = "page content line\n" * 200
        c.extract_cache_store(["https://example.com"], content, db_path=db)
        import sqlite3, sqlite_vec as sv
        conn = sqlite3.connect(str(db))
        conn.enable_load_extension(True); sv.load(conn); conn.enable_load_extension(False)
        row = conn.execute("SELECT result_blob FROM extract_meta").fetchone()
        conn.close()
        blob = row[0]
        assert isinstance(blob, bytes)
        assert len(blob) < len(content)
        assert zlib.decompress(blob).decode() == content

    def test_created_at_preserved(self, db):
        before = time.time()
        c.extract_cache_store(["https://example.com"], "content", db_path=db)
        after = time.time()
        hit = c.extract_cache_lookup(["https://example.com"], db_path=db)
        assert before <= hit.created_at <= after

    def test_similarity_is_one_for_exact(self, db):
        c.extract_cache_store(["https://example.com"], "content", db_path=db)
        hit = c.extract_cache_lookup(["https://example.com"], db_path=db)
        assert hit.similarity == 1.0


# ---------------------------------------------------------------------------
# Stats and list_recent
# ---------------------------------------------------------------------------

class TestCacheStats:
    def test_empty_db_stats(self, db):
        stats = c.cache_stats(db_path=db)
        assert stats["search_count"] == 0
        assert stats["extract_count"] == 0
        assert stats["search_bytes"] == 0
        assert stats["extract_bytes"] == 0

    def test_counts_increment(self, db):
        c.search_cache_store("query one unique words", "r1", db_path=db)
        c.search_cache_store("query two unique words", "r2", db_path=db)
        c.extract_cache_store(["https://a.com"], "content", db_path=db)
        stats = c.cache_stats(db_path=db)
        assert stats["search_count"] == 2
        assert stats["extract_count"] == 1

    def test_bytes_nonzero_after_store(self, db):
        c.search_cache_store("bytes test query", "some content here", db_path=db)
        assert c.cache_stats(db_path=db)["search_bytes"] > 0

    def test_list_recent_returns_both_tables(self, db):
        c.search_cache_store("recent query words", "result", db_path=db)
        c.extract_cache_store(["https://example.com"], "content", db_path=db)
        recent = c.list_recent(limit=5, db_path=db)
        assert len(recent["searches"]) == 1
        assert recent["searches"][0]["query"] == "recent query words"
        assert len(recent["extracts"]) == 1

    def test_list_recent_respects_limit(self, db):
        for i in range(10):
            c.search_cache_store(f"query number {i} unique terms here", f"r{i}", db_path=db)
        recent = c.list_recent(limit=3, db_path=db)
        assert len(recent["searches"]) == 3

    def test_list_recent_sorted_newest_first(self, db):
        c.search_cache_store("first stored query unique", "r1", db_path=db)
        time.sleep(0.02)
        c.search_cache_store("second stored query unique", "r2", db_path=db)
        recent = c.list_recent(limit=10, db_path=db)
        queries = [r["query"] for r in recent["searches"]]
        assert queries[0] == "second stored query unique"
        assert queries[1] == "first stored query unique"
