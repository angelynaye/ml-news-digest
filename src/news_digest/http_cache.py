"""Disk-backed HTTP response cache.

Keyed by URL hash. TTL applied on read (default 6h). Stores the raw response
body so feedparser/json can re-parse identically. Used by fetcher.py,
arxiv_authors.py, huggingface.py, and reddit.py to avoid re-hitting the same
upstreams multiple times per day (and to survive rate-limit hiccups).
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger(__name__)

_DEFAULT_DIR = Path(os.environ.get("NEWS_DIGEST_CACHE_DIR", "data/http_cache"))


def _key(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()


def get_cached(url: str, *, ttl_sec: int, cache_dir: Path | None = None) -> bytes | None:
    d = cache_dir or _DEFAULT_DIR
    meta_path = d / f"{_key(url)}.meta.json"
    body_path = d / f"{_key(url)}.body"
    if not meta_path.exists() or not body_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        fetched_at = datetime.fromisoformat(meta["fetched_at"])
        age = (datetime.now(timezone.utc) - fetched_at).total_seconds()
        if age > ttl_sec:
            return None
        return body_path.read_bytes()
    except Exception:
        return None


def put_cached(url: str, body: bytes, *, status: int = 200, cache_dir: Path | None = None) -> None:
    d = cache_dir or _DEFAULT_DIR
    d.mkdir(parents=True, exist_ok=True)
    meta_path = d / f"{_key(url)}.meta.json"
    body_path = d / f"{_key(url)}.body"
    meta = {
        "url": url, "status": status,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    body_path.write_bytes(body)
    meta_path.write_text(json.dumps(meta))


def cached_get(url: str, *, ttl_sec: int = 21600, headers: dict | None = None,
               timeout: int = 15, cache_dir: Path | None = None) -> bytes | None:
    """GET with disk cache. Returns body bytes or None on error."""
    cached = get_cached(url, ttl_sec=ttl_sec, cache_dir=cache_dir)
    if cached is not None:
        log.debug("cache hit %s", url)
        return cached
    try:
        r = requests.get(url, headers=headers or {}, timeout=timeout)
        r.raise_for_status()
        put_cached(url, r.content, status=r.status_code, cache_dir=cache_dir)
        return r.content
    except Exception as e:
        log.warning("cached_get failed for %s: %s", url, e)
        return None


def prune(max_age_days: int = 7, cache_dir: Path | None = None) -> int:
    d = cache_dir or _DEFAULT_DIR
    if not d.exists():
        return 0
    cutoff = datetime.now(timezone.utc).timestamp() - max_age_days * 86400
    n = 0
    for p in d.glob("*"):
        if p.stat().st_mtime < cutoff:
            try: p.unlink(); n += 1
            except Exception: pass
    return n
