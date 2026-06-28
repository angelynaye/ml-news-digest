"""Preference learning from clicks + thumbs feedback.

Storage:
  data/feedback.jsonl  — append-only log of {ts, kind, item_id, url, source, title, score}
  data/preferences.md  — human-readable summary, auto-updated; user can also hand-edit
                         to add explicit preferences ("more X", "less Y").

Signal flow:
  - server.py appends to feedback.jsonl on /click /up /down.
  - On each digest run, compute_lifts() reads the jsonl and produces a dict of
    {keyword: lift, source: lift, domain: lift} that the ranker multiplies in.
  - regenerate_preferences_md() then rewrites preferences.md so the user can read
    what the system thinks they like.
"""
from __future__ import annotations
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

log = logging.getLogger(__name__)

# Vote weights — tuned to be present but not dominant.
# A single click adds ~3% to the matched dimensions' multiplier; a thumbs-up
# adds ~15%; downvote subtracts ~15%. Many votes over weeks accumulate up to
# the per-dimension cap (LIFT_CAP_HIGH / LIFT_CAP_LOW) defined below.
WEIGHTS = {"click": 0.3, "up": 1.5, "down": -1.5}

# Multiplier shape: m = 1.0 + LIFT_SLOPE * weighted_signal, clamped to range.
LIFT_SLOPE = 0.10
LIFT_CAP_HIGH = 1.8     # at most 1.8× boost per dimension
LIFT_CAP_LOW  = 0.6     # at most 0.6× penalty per dimension


def append_event(feedback_path: Path, event: dict) -> None:
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with feedback_path.open("a") as f:
        f.write(json.dumps(event) + "\n")


def _load_events(feedback_path: Path, days: int = 60) -> list[dict]:
    if not feedback_path.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    events: list[dict] = []
    for line in feedback_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
            ts = datetime.fromisoformat(e["ts"])
            if ts < cutoff:
                continue
            events.append(e)
        except Exception:
            pass
    return events


_STOP = {
    "the","a","and","or","of","to","for","with","in","on","at","is","are","be",
    "this","that","from","by","an","as","it","its","was","were","has","have",
    "new","why","how","what","using","use","via","into","over","under","ai",
    "ml","model","models","paper","papers","news","blog","update","release",
}


def _tokens(text: str) -> list[str]:
    out = []
    for w in (text or "").lower().split():
        w = "".join(c for c in w if c.isalnum() or c == "-")
        if len(w) >= 3 and w not in _STOP and not w.isdigit():
            out.append(w)
    return out


def compute_lifts(feedback_path: Path) -> dict:
    """Return per-keyword/source/domain multiplicative score modifiers.

    Modifier = 1.0 + 0.25 * net_weight; clamped to [0.3, 2.5].
    """
    events = _load_events(feedback_path)
    if not events:
        return {"keywords": {}, "sources": {}, "domains": {}}

    kw_score: Counter[str] = Counter()
    src_score: Counter[str] = Counter()
    dom_score: Counter[str] = Counter()
    for e in events:
        w = WEIGHTS.get(e.get("kind", ""), 0)
        if not w:
            continue
        title = e.get("title", "")
        for t in _tokens(title):
            kw_score[t] += w
        src = e.get("source")
        if src: src_score[src] += w
        url = e.get("url", "")
        if url:
            try:
                d = urlparse(url).netloc.lower()
                if d.startswith("www."): d = d[4:]
                if d: dom_score[d] += w
            except Exception:
                pass

    def to_mult(c: Counter) -> dict:
        out = {}
        for k, v in c.items():
            m = 1.0 + LIFT_SLOPE * v
            out[k] = max(LIFT_CAP_LOW, min(LIFT_CAP_HIGH, m))
        return out

    return {
        "keywords": to_mult(kw_score),
        "sources":  to_mult(src_score),
        "domains":  to_mult(dom_score),
    }


def apply_lifts(items: list[dict], lifts: dict) -> None:
    """Multiplies item['score'] in place by the geometric mean of matching lifts."""
    kw = lifts.get("keywords", {})
    src = lifts.get("sources", {})
    dom = lifts.get("domains", {})
    if not (kw or src or dom):
        return
    for it in items:
        mult = 1.0
        # Source lift (exact match).
        if it["source"] in src:
            mult *= src[it["source"]]
        # Domain lift.
        try:
            d = urlparse(it["url"]).netloc.lower()
            if d.startswith("www."): d = d[4:]
            if d in dom:
                mult *= dom[d]
        except Exception:
            pass
        # Keyword lift — take top 3 matched keywords (was 5) and dampen further
        # so a single popular term can't quintuple the score.
        toks = _tokens(it.get("title", ""))
        matched = [kw[t] for t in toks if t in kw]
        if matched:
            matched.sort(key=lambda v: -abs(v - 1.0))
            for v in matched[:3]:
                mult *= (1.0 + (v - 1.0) * 0.25)
        # Final cap on the combined item multiplier so feedback can't dominate
        # raw keyword/recency scoring.
        it["score"] *= max(0.4, min(2.2, mult))


def regenerate_preferences_md(feedback_path: Path, md_path: Path) -> None:
    events = _load_events(feedback_path, days=90)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    if not events:
        md_path.write_text(
            "# News-digest preferences\n\n"
            "_No feedback yet. Click links or use 👍/👎 in your digest to teach the ranker._\n\n"
            "## Manual hints (optional)\n\n"
            "You can also add lines below — the ranker will pick them up:\n\n"
            "    BOOST: vllm, paged attention, fp8\n"
            "    SUPPRESS: blockchain, web3\n"
        )
        return

    lifts = compute_lifts(feedback_path)
    def top(d: dict, n: int, reverse: bool = True) -> list:
        return sorted(d.items(), key=lambda kv: -kv[1] if reverse else kv[1])[:n]

    lines = [
        "# News-digest preferences",
        "",
        f"_Auto-generated from {len(events)} feedback events. Edit the **Manual hints** block at the bottom; the rest is overwritten on each run._",
        "",
        "## Sources you click most",
        "",
    ]
    for src, mult in top(lifts["sources"], 8):
        lines.append(f"- **{src}** — boost ×{mult:.2f}")
    lines += ["", "## Keywords you click most", ""]
    boosted_kw = [(k, v) for k, v in lifts["keywords"].items() if v > 1.0]
    suppressed_kw = [(k, v) for k, v in lifts["keywords"].items() if v < 1.0]
    for k, v in sorted(boosted_kw, key=lambda kv: -kv[1])[:15]:
        lines.append(f"- `{k}` ×{v:.2f}")
    if suppressed_kw:
        lines += ["", "## Keywords you dismiss", ""]
        for k, v in sorted(suppressed_kw, key=lambda kv: kv[1])[:10]:
            lines.append(f"- `{k}` ×{v:.2f}")
    lines += [
        "",
        "---",
        "",
        "## Manual hints (optional)",
        "",
        "Add lines like the following — they survive auto-regeneration:",
        "",
        "    <!-- MANUAL_HINTS_START -->",
        "    BOOST: vllm, paged attention, fp8",
        "    SUPPRESS: blockchain, web3, NFT",
        "    <!-- MANUAL_HINTS_END -->",
        "",
    ]
    # Preserve any existing manual hints block.
    if md_path.exists():
        prev = md_path.read_text()
        start = prev.find("<!-- MANUAL_HINTS_START -->")
        end = prev.find("<!-- MANUAL_HINTS_END -->")
        if start != -1 and end != -1 and end > start:
            block = prev[start: end + len("<!-- MANUAL_HINTS_END -->")]
            # Replace the placeholder example with the existing block.
            joined = "\n".join(lines)
            joined = joined.replace(
                "    <!-- MANUAL_HINTS_START -->\n    BOOST: vllm, paged attention, fp8\n    SUPPRESS: blockchain, web3, NFT\n    <!-- MANUAL_HINTS_END -->",
                block,
            )
            md_path.write_text(joined)
            return
    md_path.write_text("\n".join(lines))


def load_manual_hints(md_path: Path) -> dict:
    """Read BOOST: / SUPPRESS: lines inside the MANUAL_HINTS block."""
    if not md_path.exists():
        return {"boost": [], "suppress": []}
    text = md_path.read_text()
    start = text.find("<!-- MANUAL_HINTS_START -->")
    end = text.find("<!-- MANUAL_HINTS_END -->")
    if start == -1 or end == -1:
        return {"boost": [], "suppress": []}
    block = text[start:end]
    boost: list[str] = []
    suppress: list[str] = []
    for raw in block.splitlines():
        line = raw.strip()
        if line.lower().startswith("boost:"):
            boost += [s.strip().lower() for s in line.split(":", 1)[1].split(",") if s.strip()]
        elif line.lower().startswith("suppress:"):
            suppress += [s.strip().lower() for s in line.split(":", 1)[1].split(",") if s.strip()]
    return {"boost": boost, "suppress": suppress}


def apply_manual_hints(items: list[dict], hints: dict) -> None:
    boost = hints.get("boost", [])
    suppress = hints.get("suppress", [])
    if not boost and not suppress:
        return
    for it in items:
        hay = (it.get("title", "") + " " + it.get("tldr", "")).lower()
        m = 1.0
        for b in boost:
            if b in hay: m *= 1.6
        for s in suppress:
            if s in hay: m *= 0.2
        it["score"] *= m
