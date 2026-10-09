"""Fetch items from RSS/Atom feeds and HTML scrapers."""
from __future__ import annotations
import hashlib
import logging
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import feedparser
import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

UA = "Mozilla/5.0 (Macintosh; news-digest/0.1; +https://example.invalid)"
HEADERS = {"User-Agent": UA, "Accept": "application/rss+xml, application/atom+xml, text/xml, */*"}
REQUEST_TIMEOUT = 15

RSSHUB_CONTAINER = "rsshub"
RSSHUB_HEALTH_URL = "http://localhost:1200/"
_rsshub_restart_lock = threading.Lock()


def _restart_rsshub_and_wait(container: str = RSSHUB_CONTAINER, wait_s: int = 60) -> bool:
    """Restart the RSSHub container and wait for its root URL to respond. Serialized across threads."""
    docker = shutil.which("docker")
    if not docker:
        log.warning("docker binary not found; cannot restart %s", container)
        return False
    with _rsshub_restart_lock:
        try:
            subprocess.run([docker, "restart", container], check=True, timeout=60, capture_output=True)
        except Exception as e:
            log.warning("docker restart %s failed: %s", container, e)
            return False
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            try:
                r = requests.get(RSSHUB_HEALTH_URL, timeout=3)
                if r.status_code < 500:
                    log.info("rsshub healthy after restart (HTTP %d)", r.status_code)
                    return True
            except Exception:
                pass
            time.sleep(2)
        log.warning("rsshub did not become healthy within %ds after restart", wait_s)
        return False


def _stable_id(url: str, title: str | None) -> str:
    h = hashlib.sha1()
    h.update((url or "").encode("utf-8"))
    h.update(b"\x00")
    h.update((title or "").encode("utf-8"))
    return h.hexdigest()


def _parse_published(entry) -> datetime | None:
    for k in ("published", "updated", "created"):
        val = entry.get(k)
        if not val:
            continue
        try:
            dt = parsedate_to_datetime(val)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            pass
    pp = entry.get("published_parsed") or entry.get("updated_parsed")
    if pp:
        try:
            return datetime(*pp[:6], tzinfo=timezone.utc)
        except Exception:
            pass
    return None


_SOURCE_KIND_MAP = {
    # Known sources → kind chip
    "DeepMind": "tech blog", "Google Research": "tech blog", "OpenAI": "tech blog",
    "Hugging Face": "tech blog", "PyTorch": "tech blog", "NVIDIA Dev": "tech blog",
    "Microsoft Research": "tech blog", "Together.ai": "tech blog", "Replicate": "tech blog",
    "Anthropic": "tech blog",
    "BAIR (Berkeley)": "academic", "Stanford AI Lab": "academic",
    "CMU ML": "academic", "MIT News (AI)": "academic",
    "Import AI": "newsletter", "Latent Space": "newsletter",
    "Ahead of AI": "newsletter", "Smol AI News": "newsletter",
    "Karpathy": "personal blog", "Lilian Weng": "personal blog",
    "Chip Huyen": "personal blog", "Simon Willison": "personal blog",
    "arXiv cs.LG": "paper", "arXiv cs.DC": "paper", "HF Daily Papers": "paper",
    "HF Models": "release",
    "Hacker News": "aggregator", "HN ML/Infra filter": "aggregator",
    "Marginal Revolution": "macro", "Stratechery": "macro", "Noahpinion": "macro",
    "Matt Stoller BIG": "macro", "WSJ Markets": "finance", "FT Alphaville": "finance",
}


def _source_kind(source: str) -> str:
    if source in _SOURCE_KIND_MAP:
        return _SOURCE_KIND_MAP[source]
    if source.startswith("@"):           return "twitter"
    if source.startswith("arXiv ·"):     return "paper"
    if source.startswith("YT ·"):        return "video"
    if source.startswith("Bilibili"):    return "video"
    if source.startswith("Zhihu"):       return "personal blog"
    if source.startswith("r/"):          return "aggregator"
    return "tech blog"


def _entry_summary(entry) -> str:
    for k in ("summary", "description"):
        if entry.get(k):
            return BeautifulSoup(entry[k], "lxml").get_text(" ", strip=True)[:1200]
    content = entry.get("content")
    if content and isinstance(content, list) and content:
        return BeautifulSoup(content[0].get("value", ""), "lxml").get_text(" ", strip=True)[:1200]
    return ""


def fetch_feed(source: dict, lookback_hours: int, cache_ttl_sec: int = 21600) -> list[dict]:
    """Fetch one RSS/Atom feed and return normalized item dicts."""
    from .http_cache import cached_get
    name = source["name"]
    url = source["url"]
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    is_bilibili = name.startswith("Bilibili")
    timeout = 90 if is_bilibili else REQUEST_TIMEOUT
    max_attempts = 3 if is_bilibili else 1
    body = None
    feed = None
    for attempt in range(1, max_attempts + 1):
        ttl = cache_ttl_sec if attempt == 1 else 0
        body = cached_get(url, ttl_sec=ttl, headers=HEADERS, timeout=timeout)
        if body is not None:
            feed = feedparser.parse(body)
            if not is_bilibili or feed.entries:
                break
        if is_bilibili and attempt < max_attempts:
            log.warning("bilibili fetch failed for %s (attempt %d/%d) — restarting rsshub",
                        name, attempt, max_attempts)
            _restart_rsshub_and_wait()
    if body is None:
        log.warning("fetch failed for %s", name)
        return []
    if feed is None:
        feed = feedparser.parse(body)
    if is_bilibili and not feed.entries:
        log.warning("bilibili feed empty for %s after %d attempts (url=%s)",
                    name, max_attempts, url)
        return []
    items: list[dict] = []
    for entry in feed.entries:
        link = entry.get("link") or ""
        title = (entry.get("title") or "").strip()
        if not link or not title:
            continue
        published = _parse_published(entry)
        # If the feed doesn't supply a date, keep the item — it'll get
        # filtered by dedup if old.
        if published and published < cutoff and not is_bilibili:
            continue
        items.append({
            "id": _stable_id(link, title),
            "url": link,
            "title": title,
            "summary": _entry_summary(entry),
            "source": name,
            "source_kind": _source_kind(name),
            "category": source.get("category", "ml_infra"),
            "weight": float(source.get("weight", 1.0)),
            "published_at": published.isoformat() if published else None,
        })
    log.info("fetched %d items from %s", len(items), name)
    return items


def _scrape_anthropic_section(url: str, href_prefix: str, source_name: str,
                              category: str, weight: float, source_kind: str = "tech blog") -> list[dict]:
    """Generic scraper for Anthropic's `/news` and `/research` listing pages."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        log.warning("anthropic scrape failed for %s: %s", url, e)
        return []
    soup = BeautifulSoup(resp.text, "lxml")
    items: list[dict] = []
    seen_urls: set[str] = set()
    for a in soup.select(f'a[href^="{href_prefix}"]'):
        href = a.get("href", "")
        if not href or href.rstrip("/") == href_prefix.rstrip("/") or href in seen_urls:
            continue
        seen_urls.add(href)
        title_el = a.find(["h2", "h3", "h4"]) or a
        title = title_el.get_text(" ", strip=True)
        if not title or len(title) < 10:
            continue
        full_url = "https://www.anthropic.com" + href
        items.append({
            "id": _stable_id(full_url, title),
            "url": full_url,
            "title": title,
            "summary": "",
            "source": source_name,
            "source_kind": source_kind,
            "category": category,
            "weight": weight,
            "published_at": None,
        })
    log.info("scraped %d items from %s", len(items), source_name)
    return items[:30]


def scrape_anthropic_news(cfg: dict, lookback_hours: int) -> list[dict]:
    if not cfg.get("enabled"):
        return []
    return _scrape_anthropic_section(
        url=cfg["url"],
        href_prefix="/news/",
        source_name="Anthropic",
        category=cfg.get("category", "ml_infra"),
        weight=float(cfg.get("weight", 1.0)),
        source_kind="tech blog",
    )


def scrape_anthropic_research(cfg: dict, lookback_hours: int) -> list[dict]:
    """Anthropic's research listing — interpretability papers etc. No RSS."""
    if not cfg.get("enabled"):
        return []
    return _scrape_anthropic_section(
        url=cfg.get("url", "https://www.anthropic.com/research"),
        href_prefix="/research/",
        source_name="Anthropic Research",
        category=cfg.get("category", "papers"),
        weight=float(cfg.get("weight", 1.6)),
        source_kind="paper",
    )


def fetch_all(sources: list[dict], lookback_hours: int, max_workers: int = 8,
              cache_ttl_sec: int = 21600) -> list[dict]:
    """Fetch all enabled sources in parallel (cached on disk for cache_ttl_sec)."""
    enabled = [s for s in sources if s.get("enabled", True)]
    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_feed, s, lookback_hours, cache_ttl_sec): s for s in enabled}
        for f in as_completed(futures):
            try:
                out.extend(f.result())
            except Exception as e:
                log.warning("worker failed: %s", e)
    return out
