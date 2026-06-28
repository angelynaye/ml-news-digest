"""Hugging Face JSON endpoints for model releases and daily papers.

These are NOT RSS — they're plain JSON — so we adapt them into the same item shape
the rest of the pipeline uses.

Endpoints (verified 2026-06):
  - https://huggingface.co/api/models?sort=trendingScore&direction=-1&limit=N
  - https://huggingface.co/api/daily_papers
"""
from __future__ import annotations
import hashlib
import logging
from datetime import datetime, timedelta, timezone

import requests

log = logging.getLogger(__name__)

UA = "Mozilla/5.0 (Macintosh; news-digest/0.1)"
HEADERS = {"User-Agent": UA, "Accept": "application/json"}
TIMEOUT = 15


def _sid(url: str, title: str) -> str:
    h = hashlib.sha1()
    h.update(url.encode()); h.update(b"\x00"); h.update(title.encode())
    return h.hexdigest()


def fetch_trending_models(cfg: dict, lookback_hours: int) -> list[dict]:
    """Pull trending models, filter by downloads/likes thresholds."""
    if not cfg.get("enabled"):
        return []
    limit = int(cfg.get("limit", 50))
    min_downloads = int(cfg.get("min_downloads", 1000))
    min_likes = int(cfg.get("min_likes", 30))
    from .http_cache import cached_get
    import json as _json
    url = f"https://huggingface.co/api/models?sort=trendingScore&direction=-1&limit={limit}"
    body = cached_get(url, ttl_sec=21600, headers=HEADERS, timeout=TIMEOUT)
    if body is None:
        log.warning("HF models fetch failed")
        return []
    try:
        data = _json.loads(body)
    except Exception as e:
        log.warning("HF models parse failed: %s", e); return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    out: list[dict] = []
    for m in data:
        likes = int(m.get("likes", 0) or 0)
        dls   = int(m.get("downloads", 0) or 0)
        # Keep if it crosses EITHER threshold (community signal or download signal).
        if likes < min_likes and dls < min_downloads:
            continue
        created = m.get("createdAt")
        dt = None
        if created:
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                if dt < cutoff:
                    continue
            except Exception:
                pass
        mid = m.get("id") or m.get("modelId")
        if not mid:
            continue
        link = f"https://huggingface.co/{mid}"
        pipeline = m.get("pipeline_tag") or ""
        title = f"HF model: {mid}" + (f" · {pipeline}" if pipeline else "")
        summary_bits = []
        if pipeline:
            summary_bits.append(f"task: {pipeline}")
        if dls:
            summary_bits.append(f"downloads: {dls:,}")
        if likes:
            summary_bits.append(f"likes: {likes}")
        tags = [t for t in (m.get("tags") or []) if not t.startswith("region:")][:6]
        if tags:
            summary_bits.append("tags: " + ", ".join(tags))
        out.append({
            "id": _sid(link, title),
            "url": link,
            "title": title,
            "summary": " · ".join(summary_bits),
            "source": "HF Models",
            "source_kind": "release",
            "category": cfg.get("category", "ml_infra"),
            "weight": float(cfg.get("weight", 1.0)),
            "published_at": dt.isoformat() if dt else None,
        })
    log.info("HF trending models: kept %d (filtered from %d)", len(out), len(data))
    return out


def fetch_daily_papers(cfg: dict) -> list[dict]:
    """HF Daily Papers — a curated daily list of arXiv papers."""
    if not cfg.get("enabled"):
        return []
    from .http_cache import cached_get
    import json as _json
    url = "https://huggingface.co/api/daily_papers"
    body = cached_get(url, ttl_sec=21600, headers=HEADERS, timeout=TIMEOUT)
    if body is None:
        log.warning("HF daily_papers fetch failed")
        return []
    try:
        data = _json.loads(body)
    except Exception as e:
        log.warning("HF daily_papers parse failed: %s", e); return []
    out: list[dict] = []
    for entry in data[: int(cfg.get("limit", 30))]:
        paper = entry.get("paper", {})
        arxiv_id = paper.get("id")
        if not arxiv_id:
            continue
        link = f"https://arxiv.org/abs/{arxiv_id}"
        title = (paper.get("title") or "").strip()
        if not title:
            continue
        summary = (paper.get("summary") or "").strip()[:1200]
        authors = [a.get("name", "") for a in (paper.get("authors") or [])][:5]
        if authors:
            summary = f"Authors: {', '.join(authors)}. {summary}"
        pub = entry.get("publishedAt") or paper.get("publishedAt")
        dt = None
        if pub:
            try:
                dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
            except Exception:
                pass
        out.append({
            "id": _sid(link, title),
            "url": link,
            "title": title,
            "summary": summary,
            "source": "HF Daily Papers",
            "source_kind": "paper",
            "category": cfg.get("category", "papers"),
            "weight": float(cfg.get("weight", 1.0)),
            "published_at": dt.isoformat() if dt else None,
        })
    log.info("HF daily papers: %d items", len(out))
    return out
