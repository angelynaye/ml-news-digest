"""Find Twitter/X discussion of a news item by web-searching for `site:x.com <terms>`.

No API. Uses DuckDuckGo's HTML interface (which permits scraping with a UA header).
Results: a hit count (fed into the ranker) + a small list of x.com URLs (rendered
inline in the digest).

Tradeoffs:
  - DuckDuckGo HTML is rate-limited per IP; we cap parallelism and back off.
  - Hit count is noisy (DDG doesn't expose totals reliably). We use the
    *number of x.com results returned* as a proxy.
  - Only the top items get this lookup (cost: 1 HTTP request each).
"""
from __future__ import annotations
import logging
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15"
HEADERS = {"User-Agent": UA, "Accept": "text/html"}
TIMEOUT = 12

# Stop words we strip from the title before searching.
_STOP = {
    "the","a","an","and","or","of","to","for","with","in","on","at","is","are",
    "be","this","that","from","by","as","it","its","was","were","has","have",
    "new","why","how","what","using","use","via","into","over","under","via",
    "introducing","announcing","release","released","launch","launches","update",
    "now","today","ai","ml","model","models","paper","blog","post","news",
}


def _key_terms(title: str, max_terms: int = 5) -> str:
    """Pull the highest-signal words/phrases from the title.

    Heuristic: keep proper nouns (capitalized after the first word), tech
    acronyms (all-caps or with digits), and any remaining content words —
    then truncate to the most informative.
    """
    # Strip trailing " — Author Name" we add for arXiv author items.
    title = re.split(r"\s+—\s+", title)[0]
    # Tokenize on word chars + hyphens.
    raw = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.]*", title)
    keep: list[str] = []
    for i, w in enumerate(raw):
        wl = w.lower()
        if wl in _STOP:
            continue
        if len(w) <= 2 and not w.isupper():
            continue
        keep.append(w)
    # Prefer terms with capitals or digits (likely proper nouns / project names).
    keep.sort(key=lambda w: (not (any(c.isupper() for c in w[1:]) or any(c.isdigit() for c in w))))
    return " ".join(keep[:max_terms])


def _extract_x_urls(html: str, max_results: int) -> list[str]:
    """Pull direct tweet URLs out of any search-result HTML page."""
    soup = BeautifulSoup(html, "lxml")
    seen: set[str] = set()
    urls: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Some engines wrap in /url?q=… or /l/?uddg=… — unwrap.
        m = re.search(r"(?:uddg|q|url)=([^&]+)", href)
        if m and m.group(1).startswith("http"):
            href = urllib.parse.unquote(m.group(1))
        if "x.com/" not in href and "twitter.com/" not in href:
            continue
        href = href.replace("twitter.com/", "x.com/").split("?")[0]
        if "/status/" not in href:
            continue
        if href in seen:
            continue
        seen.add(href)
        urls.append(href)
        if len(urls) >= max_results:
            break
    return urls


def _ddg_search(query: str, max_results: int = 5) -> list[str]:
    q = f"site:x.com OR site:twitter.com {query}"
    url = "https://html.duckduckgo.com/html/"
    try:
        r = requests.post(url, data={"q": q}, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return _extract_x_urls(r.text, max_results)
    except Exception as e:
        log.info("ddg search failed for %r: %s", query, e)
        return []


def _bing_search(query: str, max_results: int = 5) -> list[str]:
    """Fallback search engine when DDG rate-limits us."""
    q = urllib.parse.quote_plus(f"site:x.com OR site:twitter.com {query}")
    url = f"https://www.bing.com/search?q={q}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return _extract_x_urls(r.text, max_results)
    except Exception as e:
        log.info("bing search failed for %r: %s", query, e)
        return []


def _search(query: str, max_results: int) -> list[str]:
    urls = _ddg_search(query, max_results)
    if urls:
        return urls
    return _bing_search(query, max_results)


def find_discussion(item: dict, max_results: int = 3) -> tuple[int, list[str]]:
    """Return (hit_count, list of x.com URLs) for one item."""
    query = _key_terms(item.get("title", ""))
    if not query.strip():
        return 0, []
    urls = _search(query, max_results=max_results)
    return len(urls), urls


def annotate_items(items: list[dict], cfg: dict) -> None:
    """For each item, attach `x_discussion_count` (int) and `x_discussion_urls` (list).

    Scored items get a multiplicative boost via the score field — see ranker call site.
    """
    if not cfg.get("enabled"):
        return
    max_workers = int(cfg.get("max_workers", 4))
    max_results = int(cfg.get("max_results_per_item", 3))
    delay = float(cfg.get("sleep_between_sec", 0.3))

    def worker(it):
        try:
            n, urls = find_discussion(it, max_results=max_results)
            it["x_discussion_count"] = n
            it["x_discussion_urls"] = urls
        except Exception as e:
            log.warning("x_discussion failed for %s: %s", it.get("url"), e)
            it["x_discussion_count"] = 0
            it["x_discussion_urls"] = []
        time.sleep(delay)  # be polite

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(worker, items))


def apply_discussion_lift(items: list[dict], cfg: dict) -> None:
    """Multiply score by 1 + weight * log(1 + hits). Cap to avoid runaway."""
    if not cfg.get("enabled"):
        return
    import math
    weight = float(cfg.get("score_weight", 0.4))
    cap = float(cfg.get("max_multiplier", 2.0))
    for it in items:
        n = int(it.get("x_discussion_count", 0))
        if n <= 0:
            continue
        mult = 1.0 + weight * math.log(1 + n)
        it["score"] = it["score"] * min(mult, cap)
