"""
Kagi query/result cache backed by SQLite + sqlite-vec.

Embeddings are generated with model2vec (potion-base-8M, 256-dim float32).
Vector similarity search is delegated entirely to sqlite-vec's vec0 virtual
table — KNN in pure SQL, no Python math loop.

Results are stored zlib-compressed to save disk space.

Schema (two pairs of tables — one for metadata, one vec0 for vectors)
----------------------------------------------------------------------
search_meta   (id, query, result_blob, created_at)
vec_search    vec0 virtual table — rowid FK → search_meta.id
              embedding float[256]

extract_meta  (id, url_key, urls_json, result_blob, created_at)
extract_vec   vec0 virtual table — rowid FK → extract_meta.id
              embedding float[256]

Notes
-----
- vec0 virtual tables don't support normal UPDATE/DELETE by rowid;
  we delete + reinsert both the meta row and the vec row when replacing.
- L2 distance threshold for a "cache hit": ≤ 0.85
  (empirically: synonym queries ~0.3–0.7, unrelated queries ~1.3+)
- The embedding model is loaded once per process (module-level singleton).
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import zlib
from pathlib import Path
from typing import Optional

import numpy as np
import sqlite_vec

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = (
    Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    / "zot"
    / "kagi_cache.db"
)

EMBEDDING_MODEL = "minishlab/potion-base-8M"
EMBEDDING_DIM   = 256

# L2 distance below which a cached result is surfaced as a hit.
DISTANCE_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# Embedding model singleton
# ---------------------------------------------------------------------------

_model = None

def _get_model():
    global _model
    if _model is None:
        from model2vec import StaticModel  # lazy import — only when first needed
        _model = StaticModel.from_pretrained(EMBEDDING_MODEL)
    return _model


def embed(text: str) -> np.ndarray:
    """Return a 256-dim float32 numpy vector for `text`."""
    return _get_model().encode([text])[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------

def compress(text: str) -> bytes:
    """UTF-8 encode then zlib-compress."""
    return zlib.compress(text.encode("utf-8"), level=9)


def decompress(blob: bytes) -> str:
    """zlib-decompress then UTF-8 decode."""
    return zlib.decompress(blob).decode("utf-8")


# ---------------------------------------------------------------------------
# Database connection & schema
# ---------------------------------------------------------------------------

def _open_db(path: Path | None = None) -> sqlite3.Connection:
    db_path = path or DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # Load sqlite-vec extension
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS search_meta (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            query       TEXT    NOT NULL,
            result_blob BLOB    NOT NULL,
            created_at  REAL    NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS vec_search
            USING vec0(embedding float[{EMBEDDING_DIM}]);

        CREATE TABLE IF NOT EXISTS extract_meta (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            url_key     TEXT    NOT NULL,
            urls_json   TEXT    NOT NULL,
            result_blob BLOB    NOT NULL,
            created_at  REAL    NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS vec_extract
            USING vec0(embedding float[{EMBEDDING_DIM}]);

        CREATE INDEX IF NOT EXISTS idx_extract_url_key
            ON extract_meta(url_key);
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# CacheHit result type
# ---------------------------------------------------------------------------

class CacheHit:
    """A cached result returned from a lookup."""
    __slots__ = ("row_id", "original_query", "result", "created_at", "distance")

    def __init__(
        self,
        row_id: int,
        original_query: str,
        result: str,
        created_at: float,
        distance: float,
    ) -> None:
        self.row_id        = row_id
        self.original_query = original_query
        self.result        = result
        self.created_at    = created_at
        self.distance      = distance

    @property
    def similarity(self) -> float:
        """Convenience: map L2 distance to a 0-1 similarity score for display."""
        return max(0.0, 1.0 - self.distance / DISTANCE_THRESHOLD)


# ---------------------------------------------------------------------------
# Search cache
# ---------------------------------------------------------------------------

def search_cache_lookup(
    query: str,
    db_path: Path | None = None,
    threshold: float = DISTANCE_THRESHOLD,
) -> Optional[CacheHit]:
    """
    Embed `query`, run a KNN search against vec_search, return the nearest
    hit whose L2 distance is ≤ threshold, or None.
    """
    query_vec = embed(query)
    conn = _open_db(db_path)
    try:
        row = conn.execute(
            """
            SELECT v.rowid, v.distance, m.query, m.result_blob, m.created_at
            FROM vec_search v
            JOIN search_meta m ON m.id = v.rowid
            WHERE v.embedding MATCH ?
              AND k = 5
            ORDER BY v.distance
            LIMIT 1
            """,
            (sqlite_vec.serialize_float32(query_vec),),
        ).fetchone()
    finally:
        conn.close()

    if row is None or row["distance"] > threshold:
        return None

    return CacheHit(
        row_id=row["rowid"],
        original_query=row["query"],
        result=decompress(row["result_blob"]),
        created_at=row["created_at"],
        distance=row["distance"],
    )


def search_cache_store(
    query: str,
    result: str,
    db_path: Path | None = None,
) -> None:
    """
    Embed `query` and persist the result, replacing any existing exact-query row.
    """
    query_vec = embed(query)
    blob = compress(result)
    conn = _open_db(db_path)
    try:
        # Remove any existing exact-query entry (meta + vec rows)
        existing = conn.execute(
            "SELECT id FROM search_meta WHERE query = ?", (query,)
        ).fetchone()
        if existing:
            old_id = existing["id"]
            conn.execute("DELETE FROM search_meta WHERE id = ?", (old_id,))
            conn.execute("DELETE FROM vec_search WHERE rowid = ?", (old_id,))

        conn.execute(
            "INSERT INTO search_meta (query, result_blob, created_at) VALUES (?, ?, ?)",
            (query, blob, time.time()),
        )
        new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO vec_search (rowid, embedding) VALUES (?, ?)",
            (new_id, sqlite_vec.serialize_float32(query_vec)),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Extract cache  (exact URL-set match — no vector search needed)
# ---------------------------------------------------------------------------

def _url_key(urls: list[str]) -> str:
    """Canonical key for a set of URLs: sorted and joined."""
    return ",".join(sorted(urls))


def extract_cache_lookup(
    urls: list[str],
    db_path: Path | None = None,
) -> Optional[CacheHit]:
    """Return the cached extract result for exactly this set of URLs, or None."""
    key = _url_key(urls)
    conn = _open_db(db_path)
    try:
        row = conn.execute(
            "SELECT id, urls_json, result_blob, created_at FROM extract_meta"
            " WHERE url_key = ? ORDER BY created_at DESC LIMIT 1",
            (key,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    stored_urls = json.loads(row["urls_json"])
    return CacheHit(
        row_id=row["id"],
        original_query=", ".join(stored_urls),
        result=decompress(row["result_blob"]),
        created_at=row["created_at"],
        distance=0.0,  # exact match
    )


def extract_cache_store(
    urls: list[str],
    result: str,
    db_path: Path | None = None,
) -> None:
    """Persist an extract result, replacing any existing entry for the same URL set."""
    key = _url_key(urls)
    blob = compress(result)
    conn = _open_db(db_path)
    try:
        conn.execute("DELETE FROM extract_meta WHERE url_key = ?", (key,))
        conn.execute(
            "INSERT INTO extract_meta (url_key, urls_json, result_blob, created_at)"
            " VALUES (?, ?, ?, ?)",
            (key, json.dumps(urls), blob, time.time()),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Utility: list recent entries  (for /kagi-cache command)
# ---------------------------------------------------------------------------

def list_recent(limit: int = 20, db_path: Path | None = None) -> dict:
    """Return dicts for recent search and extract entries."""
    conn = _open_db(db_path)
    try:
        searches = conn.execute(
            "SELECT id, query, created_at, length(result_blob) AS compressed_size"
            " FROM search_meta ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        extracts = conn.execute(
            "SELECT id, urls_json, created_at, length(result_blob) AS compressed_size"
            " FROM extract_meta ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    return {
        "searches": [dict(r) for r in searches],
        "extracts": [dict(r) for r in extracts],
    }


def cache_stats(db_path: Path | None = None) -> dict:
    """Return counts and total compressed size for both tables."""
    conn = _open_db(db_path)
    try:
        s = conn.execute(
            "SELECT COUNT(*) AS count,"
            " COALESCE(SUM(length(result_blob)), 0) AS bytes"
            " FROM search_meta"
        ).fetchone()
        e = conn.execute(
            "SELECT COUNT(*) AS count,"
            " COALESCE(SUM(length(result_blob)), 0) AS bytes"
            " FROM extract_meta"
        ).fetchone()
    finally:
        conn.close()

    return {
        "search_count":  s["count"],
        "search_bytes":  s["bytes"],
        "extract_count": e["count"],
        "extract_bytes": e["bytes"],
    }
