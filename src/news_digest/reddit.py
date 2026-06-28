"""Reddit ingestion via per-subreddit `top.rss?t=day`.

JSON endpoints are 403'd to unauthenticated requests in 2026; RSS still works
but is heavily rate-limited per IP. We spread requests 10s apart with
exponential backoff on 429s.
"""
from __future__ import annotations
import hashlib
import logging
import time
from datetime import datetime, timezone

import feedparser
import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)


def _sid(url: str, title: str) -> str:
    h = hashlib.sha1(); h.update(url.encode()); h.update(b"\x00"); h.update(title.encode())
    return h.hexdigest()


def _category_for_sub(sub: str) -> str:
    sub = sub.lower()
    if sub in ("wallstreetbets", "stocks", "investing", "wallstreetbetsogs"):
        return "finance"
    return "discussions"


def fetch_subreddit(sub: str, *, cfg: dict, weight: float) -> list[dict]:
    from .http_cache import cached_get, get_cached
    url = cfg["url_template"].format(sub=sub)
    ua  = cfg.get("user_agent", "Mozilla/5.0 news-digest/0.1")
    backoff = cfg.get("retry_on_429", {}).get("backoff_seconds", [15, 30, 60, 120])
    max_attempts = int(cfg.get("retry_on_429", {}).get("max_attempts", 3))
    ttl = int(cfg.get("cache_ttl_sec", 21600))

    # Try cache first.
    body = get_cached(url, ttl_sec=ttl)
    if body is None:
        for attempt in range(max_attempts):
            try:
                r = requests.get(url, headers={"User-Agent": ua}, timeout=15)
                if r.status_code == 429:
                    wait = backoff[min(attempt, len(backoff) - 1)]
                    log.info("reddit 429 for r/%s, attempt %d/%d, sleeping %ds",
                             sub, attempt + 1, max_attempts, wait)
                    time.sleep(wait); continue
                r.raise_for_status()
                body = r.content
                # Cache for future runs.
                from .http_cache import put_cached
                put_cached(url, body, status=r.status_code)
                break
            except requests.HTTPError as e:
                if attempt == max_attempts - 1:
                    log.warning("reddit fetch failed for r/%s: %s", sub, e)
                    return []
                time.sleep(backoff[min(attempt, len(backoff) - 1)])
            except Exception as e:
                log.warning("reddit fetch failed for r/%s: %s", sub, e)
                return []
        else:
            return []

    feed = feedparser.parse(body)
    out: list[dict] = []
    for entry in feed.entries:
        link = entry.get("link") or ""
        title = (entry.get("title") or "").strip()
        if not link or not title:
            continue
        dt = None
        pp = entry.get("published_parsed") or entry.get("updated_parsed")
        if pp:
            try:
                dt = datetime(*pp[:6], tzinfo=timezone.utc)
            except Exception:
                pass
        summary_html = entry.get("summary") or ""
        summary = BeautifulSoup(summary_html, "lxml").get_text(" ", strip=True)[:1200] if summary_html else ""
        out.append({
            "id": _sid(link, title),
            "url": link,
            "title": title,
            "summary": summary,
            "source": f"r/{sub}",
            "source_kind": "aggregator",
            "category": _category_for_sub(sub),
            "weight": weight,
            "published_at": dt.isoformat() if dt else None,
        })
    return out


def fetch_all_reddit(cfg: dict) -> list[dict]:
    if not cfg.get("enabled", True):
        return []
    subs = cfg.get("subreddits", [])
    spacing = float(cfg.get("request_spacing_seconds", 10))
    weight = float(cfg.get("weight", 0.9))
    out: list[dict] = []
    for i, sub in enumerate(subs):
        items = fetch_subreddit(sub, cfg=cfg, weight=weight)
        log.info("reddit r/%s: %d items", sub, len(items))
        out.extend(items)
        if i < len(subs) - 1:
            time.sleep(spacing)
    return out
