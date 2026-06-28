"""arXiv Atom API for tracking papers by named authors.

arXiv asks for ≥3s between calls — we serialize.
"""
from __future__ import annotations
import hashlib
import logging
import time
from datetime import datetime, timedelta, timezone

import feedparser
import requests

log = logging.getLogger(__name__)

UA = "news-digest/0.1 (https://example.invalid)"
HEADERS = {"User-Agent": UA}
TIMEOUT = 20
SLEEP_BETWEEN = 3.5  # arXiv rate-limit etiquette


def _sid(url: str, title: str) -> str:
    h = hashlib.sha1(); h.update(url.encode()); h.update(b"\x00"); h.update(title.encode())
    return h.hexdigest()


def fetch_author(query: str, *, max_results: int, lookback_days: int, weight: float, name: str, category: str) -> list[dict]:
    """`query` is already URL-encoded (e.g. `%22Tri+Dao%22`)."""
    from .http_cache import cached_get
    url = (
        f"http://export.arxiv.org/api/query?search_query=au:{query}"
        f"&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    )
    body = cached_get(url, ttl_sec=21600, headers=HEADERS, timeout=TIMEOUT)
    if body is None:
        log.warning("arxiv author fetch failed for %s", name)
        return []
    feed = feedparser.parse(body)
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    out: list[dict] = []
    for e in feed.entries:
        link = e.get("link") or ""
        title = (e.get("title") or "").strip().replace("\n", " ").replace("  ", " ")
        if not link or not title:
            continue
        dt = None
        pp = e.get("published_parsed") or e.get("updated_parsed")
        if pp:
            try:
                dt = datetime(*pp[:6], tzinfo=timezone.utc)
            except Exception:
                pass
        if dt and dt < cutoff:
            continue
        summary = (e.get("summary") or "").strip()[:1200].replace("\n", " ")
        out.append({
            "id": _sid(link, title),
            "url": link,
            "title": f"{title} — {name}",
            "summary": summary,
            "source": f"arXiv · {name}",
            "source_kind": "paper",
            "category": category,
            "weight": weight,
            "published_at": dt.isoformat() if dt else None,
        })
    return out


def fetch_all_authors(cfg: dict) -> list[dict]:
    if not cfg.get("enabled"):
        return []
    out: list[dict] = []
    for a in cfg.get("authors", []):
        items = fetch_author(
            a["query"],
            max_results=int(cfg.get("max_results_per_author", 5)),
            lookback_days=int(cfg.get("lookback_days", 30)),
            weight=float(cfg.get("weight", 1.0)),
            name=a["name"],
            category=cfg.get("category", "papers"),
        )
        log.info("arxiv author %s: %d items", a["name"], len(items))
        out.extend(items)
        time.sleep(SLEEP_BETWEEN)
    return out
