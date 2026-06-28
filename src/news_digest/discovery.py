"""Weekly dynamic source discovery.

Two channels:
  1. HN domain mining — top-stories with score >= threshold, count domain frequency
     over a rolling window. Surfaces blogs that suddenly start trending.
  2. OpenAlex — top-cited recent ML papers' author affiliations / domains.

Output is a YAML file at `data/discovered_sources.yaml` that the user can
review and merge into config.yaml. We deliberately do NOT auto-add — discovery
is opt-in noise prevention.
"""
from __future__ import annotations
import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml

log = logging.getLogger(__name__)

UA = "Mozilla/5.0 (Macintosh; news-digest-discover/0.1)"
HEADERS = {"User-Agent": UA}
TIMEOUT = 15

# Domains already in default config — don't suggest them.
KNOWN_DOMAINS = {
    "anthropic.com", "deepmind.google", "research.google", "openai.com",
    "huggingface.co", "pytorch.org", "developer.nvidia.com", "microsoft.com",
    "together.ai", "replicate.com", "bair.berkeley.edu", "ai.stanford.edu",
    "blog.ml.cmu.edu", "news.mit.edu", "karpathy.github.io", "lilianweng.github.io",
    "huyenchip.com", "simonwillison.net", "importai.substack.com",
    "latent.space", "magazine.sebastianraschka.com", "marginalrevolution.com",
    "stratechery.com", "noahpinion.blog", "thebignewsletter.com",
    "dowjones.io", "ft.com", "arxiv.org", "ycombinator.com", "news.smol.ai",
    "github.com", "twitter.com", "x.com", "reddit.com", "medium.com",
    "wikipedia.org", "youtube.com",
}


def _domain(url: str) -> str | None:
    try:
        d = urlparse(url).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return None


def discover_hn_domains(min_score: int = 200, days: int = 7, max_stories: int = 300) -> list[dict]:
    """Mine HN top stories from the last N days; surface frequently-cited domains."""
    try:
        ids = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json",
                           headers=HEADERS, timeout=TIMEOUT).json()
    except Exception as e:
        log.warning("HN topstories fetch failed: %s", e)
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    counter: Counter[str] = Counter()
    examples: dict[str, str] = {}
    for sid in ids[:max_stories]:
        try:
            item = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{sid}.json",
                                headers=HEADERS, timeout=TIMEOUT).json()
        except Exception:
            continue
        if not item or item.get("type") != "story":
            continue
        url = item.get("url")
        if not url:
            continue
        ts = datetime.fromtimestamp(item.get("time", 0), tz=timezone.utc)
        if ts < cutoff:
            continue
        score = int(item.get("score", 0))
        if score < min_score:
            continue
        d = _domain(url)
        if not d or d in KNOWN_DOMAINS:
            continue
        # Skip noisy domains (news sites with too-generic content).
        if any(n in d for n in ("nytimes.com", "wsj.com", "bbc.com", "cnn.com", "theverge.com")):
            continue
        counter[d] += 1
        examples.setdefault(d, url)
    suggestions: list[dict] = []
    for d, n in counter.most_common(20):
        if n < 2:  # appeared at least twice in top stories
            continue
        suggestions.append({"domain": d, "hits": n, "example": examples[d]})
    return suggestions


def write_suggestions(suggestions: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "Domains repeatedly trending on HN that aren't in config. Review and copy useful ones into config.yaml under `sources:` (probe for /feed, /rss, /index.xml).",
        "candidates": suggestions,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    log.info("wrote %d discovery candidates to %s", len(suggestions), path)


def run_discovery(out_path: str = "data/discovered_sources.yaml") -> dict:
    sugg = discover_hn_domains()
    write_suggestions(sugg, Path(out_path))
    return {"hn_domains": len(sugg), "path": out_path}
