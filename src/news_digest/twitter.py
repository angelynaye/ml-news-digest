"""Twitter / X ingestion with four fallback strategies.

Resolution order at runtime:
  1. X API v2 (if TWITTER_BEARER_TOKEN env is set) — reliable, paid.
  2. syndication.twitter.com (no auth, returns embed JSON) — most practical free path.
  3. RSSHub (config.twitter.rsshub.base_url) — works if self-hosted.
  4. Nitter instance pool with rotation + retries — community-run, brittle.
"""
from __future__ import annotations
import hashlib
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import feedparser
import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15"
HEADERS = {"User-Agent": UA, "Accept": "application/rss+xml, application/atom+xml, text/xml, text/html, */*"}


def _sid(url: str, title: str) -> str:
    h = hashlib.sha1()
    h.update(url.encode("utf-8")); h.update(b"\x00"); h.update(title.encode("utf-8"))
    return h.hexdigest()


def _item(*, url: str, title: str, handle: str, weight: float, published: datetime | None) -> dict:
    return {
        "id": _sid(url, title),
        "url": url,
        "title": title,
        "summary": "",
        "source": f"@{handle}",
        "source_kind": "twitter",
        "category": "twitter",
        "weight": weight,
        "published_at": published.isoformat() if published else None,
    }


# ---------------- Strategy 1: X API v2 ----------------

def _x_api_recent(handle: str, since: datetime, max_items: int, bearer: str) -> list[dict]:
    """Use the X v2 'users/by/username/{}/tweets' endpoint. Requires Basic tier."""
    base = "https://api.twitter.com/2"
    h = {"Authorization": f"Bearer {bearer}"}
    try:
        u = requests.get(f"{base}/users/by/username/{handle}", headers=h, timeout=10)
        u.raise_for_status()
        uid = u.json()["data"]["id"]
    except Exception as e:
        log.warning("X API lookup failed for @%s: %s", handle, e)
        return []
    params = {
        "max_results": max(5, min(max_items, 100)),
        "tweet.fields": "created_at,text,public_metrics",
        "exclude": "retweets,replies",
        "start_time": since.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    try:
        r = requests.get(f"{base}/users/{uid}/tweets", headers=h, params=params, timeout=15)
        r.raise_for_status()
        data = r.json().get("data", [])
    except Exception as e:
        log.warning("X API tweets failed for @%s: %s", handle, e)
        return []
    out: list[dict] = []
    for t in data[:max_items]:
        text = t.get("text", "").strip()
        if not text:
            continue
        tid = t["id"]
        url = f"https://x.com/{handle}/status/{tid}"
        try:
            dt = datetime.fromisoformat(t["created_at"].replace("Z", "+00:00"))
        except Exception:
            dt = None
        out.append(_item(url=url, title=text[:280], handle=handle, weight=1.0, published=dt))
    return out


# ---------------- Strategy 2: syndication.twitter.com (no auth) ----------------
#
# The endpoint used to power embedded tweet widgets. Returns JSON of recent tweets.
# No auth, no key, no Twitter app. Heavily relied on by tools like Nitter v2.
# Caveats:
#   - Schema can change without notice.
#   - Pinned tweets and replies are included.
#   - The endpoint expects a specific `&token=` value; the simple integer below is
#     what twitter.com's own embed code generates.

def _syndication(handle: str, since: datetime, max_items: int) -> list[dict]:
    base = "https://syndication.twitter.com/srv/timeline-profile/screen-name"
    url = f"{base}/{handle}"
    params = {
        "showReplies": "false",
        "hideQuotes": "false",
        "showRetweets": "false",
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        text = r.text
    except Exception as e:
        log.warning("syndication failed for @%s: %s", handle, e)
        return []
    # The response is an HTML page with a __NEXT_DATA__ JSON blob.
    import re as _re
    m = _re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', text, _re.DOTALL)
    if not m:
        log.info("syndication returned no __NEXT_DATA__ for @%s", handle)
        return []
    import json as _json
    try:
        blob = _json.loads(m.group(1))
        entries = (
            blob.get("props", {})
                .get("pageProps", {})
                .get("timeline", {})
                .get("entries", [])
        )
    except Exception as e:
        log.warning("syndication JSON parse failed for @%s: %s", handle, e)
        return []
    out: list[dict] = []
    for entry in entries[:max_items * 3]:  # over-fetch since replies sometimes slip through
        content = entry.get("content", {}) or {}
        tweet = (content.get("tweet")
                 or content.get("itemContent", {}).get("tweet_results", {}).get("result", {})
                 or {})
        text_full = (tweet.get("full_text")
                     or tweet.get("text")
                     or tweet.get("legacy", {}).get("full_text", ""))
        text_full = text_full.strip()
        if not text_full:
            continue
        tid = (tweet.get("id_str")
               or tweet.get("rest_id")
               or tweet.get("legacy", {}).get("id_str", ""))
        if not tid:
            continue
        created = (tweet.get("created_at")
                   or tweet.get("legacy", {}).get("created_at"))
        dt = None
        if created:
            try:
                # Twitter's format: "Wed Jun 25 17:30:00 +0000 2026"
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(created)
                if dt and dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except Exception:
                pass
        if dt and dt < since:
            continue
        link = f"https://x.com/{handle}/status/{tid}"
        out.append(_item(url=link, title=text_full[:280], handle=handle, weight=1.0, published=dt))
        if len(out) >= max_items:
            break
    return out


# ---------------- Strategy 3: RSSHub ----------------

def _rsshub(handle: str, since: datetime, max_items: int, base_url: str, access_key: str | None) -> list[dict]:
    url = f"{base_url.rstrip('/')}/twitter/user/{handle}"
    params = {"limit": max_items}
    if access_key:
        params["key"] = access_key
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except Exception as e:
        log.warning("RSSHub failed for @%s at %s: %s", handle, base_url, e)
        return []
    feed = feedparser.parse(r.content)
    out: list[dict] = []
    for e in feed.entries[:max_items]:
        link = e.get("link") or ""
        if not link:
            continue
        title = BeautifulSoup(e.get("title") or e.get("summary", ""), "lxml").get_text(" ", strip=True)[:280]
        if not title:
            continue
        dt = None
        pp = e.get("published_parsed") or e.get("updated_parsed")
        if pp:
            try:
                dt = datetime(*pp[:6], tzinfo=timezone.utc)
            except Exception:
                pass
        if dt and dt < since:
            continue
        out.append(_item(url=link, title=title, handle=handle, weight=1.0, published=dt))
    return out


# ---------------- Strategy 4: Nitter pool ----------------

def _nitter_one(handle: str, since: datetime, max_items: int, instance: str, timeout: int) -> list[dict]:
    url = f"{instance.rstrip('/')}/{handle}/rss"
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    feed = feedparser.parse(r.content)
    out: list[dict] = []
    for e in feed.entries[:max_items]:
        link = (e.get("link") or "").replace(instance.rstrip("/"), "https://x.com")
        if not link:
            continue
        title = BeautifulSoup(e.get("title") or "", "lxml").get_text(" ", strip=True)[:280]
        if not title:
            continue
        dt = None
        pp = e.get("published_parsed") or e.get("updated_parsed")
        if pp:
            try:
                dt = datetime(*pp[:6], tzinfo=timezone.utc)
            except Exception:
                pass
        if dt and dt < since:
            continue
        out.append(_item(url=link, title=title, handle=handle, weight=1.0, published=dt))
    return out


def _nitter(handle: str, since: datetime, max_items: int, cfg: dict) -> list[dict]:
    instances = list(cfg.get("instances", []))
    random.shuffle(instances)
    retries = int(cfg.get("max_retries_per_instance", 2))
    timeout = int(cfg.get("request_timeout_sec", 12))
    last_err: Exception | None = None
    for inst in instances:
        for attempt in range(retries):
            try:
                items = _nitter_one(handle, since, max_items, inst, timeout)
                # 200 with 0 items usually means silent rate-limit or unknown account —
                # try other instances before giving up so one bad instance can't shadow real data.
                if items:
                    log.info("nitter@%s ok for @%s (%d items)", inst, handle, len(items))
                    return items
                log.info("nitter@%s returned 0 items for @%s — trying next instance", inst, handle)
                break  # don't retry same instance on empty
            except Exception as e:
                last_err = e
                time.sleep(0.6 * (attempt + 1))
        else:
            log.warning("nitter @%s failed for @%s: %s", inst, handle, last_err)
    return []


# ---------------- Orchestrator ----------------

def fetch_handle(handle: str, *, lookback_hours: int, max_items: int, weight: float, cfg: dict) -> list[dict]:
    since = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    bearer = os.environ.get("TWITTER_BEARER_TOKEN")

    if cfg.get("x_api", {}).get("enabled", True) and bearer:
        items = _x_api_recent(handle, since, max_items, bearer)
        if items:
            for i in items: i["weight"] = weight
            return items

    if cfg.get("syndication", {}).get("enabled", True):
        items = _syndication(handle, since, max_items)
        if items:
            log.info("syndication ok for @%s (%d items)", handle, len(items))
            for i in items: i["weight"] = weight
            return items

    rcfg = cfg.get("rsshub", {})
    if rcfg.get("enabled", True) and rcfg.get("base_url"):
        items = _rsshub(handle, since, max_items, rcfg["base_url"], rcfg.get("access_key"))
        if items:
            for i in items: i["weight"] = weight
            return items

    ncfg = cfg.get("nitter", {})
    if ncfg.get("enabled", True) and ncfg.get("instances"):
        items = _nitter(handle, since, max_items, ncfg)
        if items:
            for i in items: i["weight"] = weight
            return items

    log.warning("no Twitter source returned items for @%s — all strategies failed", handle)
    return []


def fetch_all_twitter(twitter_cfg: dict) -> list[dict]:
    if not twitter_cfg.get("enabled"):
        return []
    handles = twitter_cfg.get("handles", [])
    lookback = int(twitter_cfg.get("lookback_hours", 24))
    max_per = int(twitter_cfg.get("max_per_handle", 5))
    weight = float(twitter_cfg.get("weight", 1.0))
    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {
            ex.submit(fetch_handle, h, lookback_hours=lookback, max_items=max_per,
                      weight=weight, cfg=twitter_cfg): h
            for h in handles
        }
        for f in as_completed(futs):
            try:
                out.extend(f.result())
            except Exception as e:
                log.warning("twitter handle failed: %s", e)
    return out
