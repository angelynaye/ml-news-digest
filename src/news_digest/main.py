"""CLI entry point.

Subcommands:
  run             Fetch, summarize, send to all recipients in config.
  preview         Same as run but write HTML to data/preview.html and skip sending.
  fetch-only      Print number of items per source (no LLM calls, no email).
  discover        Mine HN for trending new domains; write data/discovered_sources.yaml.
  serve           Run the local feedback / audio Flask server.
  list-sources
  add-recipient EMAIL
  remove-recipient EMAIL
  test-smtp
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

from .fetcher import fetch_all, scrape_anthropic_news, scrape_anthropic_research
from .twitter import fetch_all_twitter
from .huggingface import fetch_trending_models, fetch_daily_papers
from .arxiv_authors import fetch_all_authors as fetch_arxiv_authors
from .discovery import run_discovery
from .personas import PERSONAS, ENGINEER, INVESTOR, score_items_by_persona, persona_for_category, apply_overrides_from_config
from .summarizer import summarize_items
from .reddit import fetch_all_reddit
from .formatter import render_html, render_text
from .mailer import send as mail_send
from .storage import Store
from .preferences import (
    compute_lifts, apply_lifts, regenerate_preferences_md,
    load_manual_hints, apply_manual_hints,
)
from .audio import generate_audio
from .twitter_discuss import annotate_items as annotate_x_discussion, apply_discussion_lift

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config.yaml"


def _ensure_aws_sso(cfg: dict) -> None:
    """Validate AWS credentials before doing anything Bedrock-related.

    Strategy:
      - Try `sts get-caller-identity` with the configured profile.
      - If that fails, run `aws sso login --profile <profile>` once.
      - If still failing, raise — don't ship a degraded digest.

    Skip entirely if the user doesn't use SSO (AWS_PROFILE unset or no `sso_session`
    in their AWS config).
    """
    import subprocess
    profile = os.environ.get("AWS_PROFILE") or cfg.get("bedrock", {}).get("aws_profile")
    if not profile:
        logging.info("no AWS_PROFILE set — skipping SSO check")
        return
    region = cfg.get("bedrock", {}).get("region") or os.environ.get("AWS_REGION", "us-east-1")

    def _identity_ok() -> bool:
        try:
            r = subprocess.run(
                ["aws", "sts", "get-caller-identity", "--profile", profile, "--region", region],
                capture_output=True, text=True, timeout=15,
            )
            return r.returncode == 0
        except Exception:
            return False

    if _identity_ok():
        logging.info("aws sso identity OK (profile=%s)", profile)
        return

    logging.warning("aws sso token expired or missing — running `aws sso login --profile %s`", profile)
    try:
        r = subprocess.run(
            ["aws", "sso", "login", "--profile", profile],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            raise SystemExit(
                f"aws sso login failed (rc={r.returncode}): {r.stderr.strip() or r.stdout.strip()}\n"
                "Run `aws sso login --profile {profile}` interactively to refresh credentials."
            )
    except subprocess.TimeoutExpired:
        raise SystemExit(
            "aws sso login timed out (browser flow did not complete in 120s). "
            f"Run `aws sso login --profile {profile}` interactively, then retry."
        )

    if not _identity_ok():
        raise SystemExit(
            f"aws sso login claimed success but `sts get-caller-identity` still fails for profile {profile}. "
            "Refusing to run the digest with broken Bedrock credentials."
        )
    logging.info("aws sso re-authenticated successfully")


def _setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"digest-{datetime.now().strftime('%Y-%m-%d')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


def load_config(path: Path | None) -> dict:
    p = path or DEFAULT_CONFIG
    if not p.exists():
        raise SystemExit(f"config not found: {p} (copy config.example.yaml to config.yaml)")
    return yaml.safe_load(p.read_text())


def save_config(cfg: dict, path: Path | None) -> None:
    p = path or DEFAULT_CONFIG
    p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))


# --------------- pipeline ---------------

def _build_executive_summary(top_items: list[dict], bedrock_cfg: dict, persona) -> dict:
    """Returns {headline: str, themes: [{label, bullets:[...]}, ...]} for one persona."""
    import boto3
    if not top_items:
        return {"headline": "", "themes": []}
    bullet_lines = []
    for it in top_items[:10]:
        tldr = it.get("tldr") or it["title"]
        bullet_lines.append(f"- [{it['source']} · {it.get('source_kind','?')}] {it['title']} — {tldr}")
    if persona.key == "investor":
        flavor = ("Optimize for what helps the reader anticipate where money is flowing in tech "
                  "and which sub-trends are gaining momentum — name specific tickers, dollar figures, "
                  "and named investors when present in the items. If signals are weak today, "
                  "say so explicitly rather than padding with generic commentary.")
    else:
        flavor = ("Optimize for the most advanced ML-infra news the reader hasn't seen yet, "
                  "including agentic tools, harnesses, courses, and infra papers. Cite specific "
                  "systems, papers, and authors. If today is slow, say so plainly.")
    prompt = (
        f"You are briefing {persona.role_description}\n\n"
        f"Top items today:\n" + "\n".join(bullet_lines) +
        f"\n\n{flavor}\n\n"
        "Return JSON with exactly this shape:\n"
        "{\n"
        '  "headline": "<10-15 words: the single biggest thread today, plain prose, no hype>",\n'
        '  "themes": [\n'
        "    {\n"
        '      "label": "<3-6 word theme name>",\n'
        '      "bullets": ["2-3 bullets, each <= 18 words, citing specific items/numbers/names"]\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Provide 2-3 themes. Be terse, no emoji, no preamble. Return ONLY the JSON."
    )
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 600,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        client = boto3.client("bedrock-runtime", region_name=bedrock_cfg.get("region", "us-east-1"))
        # Executive summary may use a different (better) model than per-item summaries.
        exec_model = bedrock_cfg.get("exec_summary_model_id") or bedrock_cfg["model_id"]
        resp = client.invoke_model(
            modelId=exec_model, body=json.dumps(body),
            contentType="application/json", accept="application/json",
        )
        payload = json.loads(resp["body"].read())
        text = payload["content"][0]["text"].strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:]
        data = json.loads(text.strip())
        return {
            "headline": (data.get("headline") or "").strip(),
            "themes": [
                {"label": (t.get("label") or "").strip(),
                 "bullets": [b.strip() for b in (t.get("bullets") or []) if b][:3]}
                for t in (data.get("themes") or [])[:3]
                if t.get("label") and t.get("bullets")
            ],
        }
    except Exception as e:
        logging.warning("executive summary failed: %s", e)
        return {"headline": "", "themes": []}


def run_pipeline(cfg: dict, *, preview: bool, fetch_only: bool) -> dict:
    # 0) Make sure AWS Bedrock credentials are good BEFORE doing work. We won't
    #    ship a degraded digest if the SSO token expired silently mid-day.
    if not fetch_only:
        _ensure_aws_sso(cfg)

    schedule  = cfg["schedule"]
    sources   = cfg.get("sources", [])
    twitter_cfg = cfg.get("twitter", {})
    anth_cfg  = cfg.get("anthropic_scraper", {})
    storage   = Store(cfg["storage"]["sqlite_path"])

    lookback = int(schedule.get("lookback_hours", 24))

    # 1) Fetch from every channel.
    rss_items     = fetch_all(sources, lookback)
    twitter_items = fetch_all_twitter(twitter_cfg)
    anth_items    = scrape_anthropic_news(anth_cfg, lookback)
    anth_research = scrape_anthropic_research(cfg.get("anthropic_research_scraper", {}), lookback)
    hf_cfg = cfg.get("huggingface", {})
    hf_models = fetch_trending_models(hf_cfg.get("trending_models", {}), lookback)
    hf_papers = fetch_daily_papers(hf_cfg.get("daily_papers", {}))
    arxiv_items = fetch_arxiv_authors(cfg.get("arxiv_authors", {}))
    reddit_items = fetch_all_reddit(cfg.get("reddit", {}))
    all_items = (rss_items + twitter_items + anth_items + anth_research +
                 hf_models + hf_papers + arxiv_items + reddit_items)
    logging.info(
        "fetched %d items (rss=%d, twitter=%d, anthropic_news=%d, anthropic_research=%d, "
        "hf_models=%d, hf_papers=%d, arxiv_authors=%d, reddit=%d)",
        len(all_items), len(rss_items), len(twitter_items),
        len(anth_items), len(anth_research),
        len(hf_models), len(hf_papers), len(arxiv_items), len(reddit_items),
    )

    # 2) Dedup against history.
    fresh = storage.filter_unseen(all_items)
    logging.info("after dedup: %d fresh items", len(fresh))

    if fetch_only:
        by_source: dict[str, int] = {}
        for it in fresh:
            by_source[it["source"]] = by_source.get(it["source"], 0) + 1
        for src, n in sorted(by_source.items(), key=lambda x: -x[1]):
            print(f"  {n:>4}  {src}")
        print(f"  ----  total {len(fresh)} fresh / {len(all_items)} fetched")
        return {"fresh": len(fresh), "fetched": len(all_items)}

    # 3) Score each item under its persona (engineer for tech, investor for finance).
    by_persona = score_items_by_persona(fresh)
    eng_top_n = int(cfg.get("ranking", {}).get("engineer_top_n", 10))
    inv_top_n = int(cfg.get("ranking", {}).get("investor_top_n", 10))

    def balanced_head(items: list[dict], target: int) -> list[dict]:
        """Round-robin across source_kinds so quotas have variety to choose from."""
        by_kind: dict[str, list[dict]] = {}
        for it in items:
            by_kind.setdefault(it.get("source_kind", "tech blog"), []).append(it)
        for v in by_kind.values():
            v.sort(key=lambda x: -x["score"])
        head: list[dict] = []; seen: set[str] = set()
        cursors = {k: 0 for k in by_kind}
        while len(head) < target:
            advanced = False
            for k in sorted(by_kind, key=lambda kk: -by_kind[kk][0]["score"] if by_kind[kk] else 0):
                arr = by_kind[k]; i = cursors[k]
                if i >= len(arr): continue
                it = arr[i]; cursors[k] += 1; advanced = True
                if it["id"] in seen: continue
                seen.add(it["id"]); head.append(it)
                if len(head) >= target: break
            if not advanced: break
        return head

    eng_head = balanced_head(by_persona["engineer"], eng_top_n * 3)
    inv_head = balanced_head(by_persona["investor"], inv_top_n * 3)
    head = eng_head + inv_head
    tech_top_n, finance_top_n = eng_top_n, inv_top_n  # for the cap stage below

    # 4) Rich summarization — each persona's items are summarized under their own prompt.
    summarize_items(eng_head, cfg.get("bedrock", {}), ENGINEER)
    summarize_items(inv_head, cfg.get("bedrock", {}), INVESTOR)

    # 4b) Probe X for discussion of each summarized item — count + URLs.
    annotate_x_discussion(head, cfg.get("x_discussion", {}))
    apply_discussion_lift(head, cfg.get("x_discussion", {}))

    # 5) Mix LLM's per-item relevance into the score, then apply feedback lifts + manual hints.
    rscore_weight = float(cfg.get("ranking", {}).get("relevance_weight", 0.6))
    for it in head:
        it["score"] = it["score"] + rscore_weight * float(it.get("relevance_score", 5))

    feedback_path  = Path(cfg["storage"].get("feedback_path",  "data/feedback.jsonl"))
    preferences_md = Path(cfg["storage"].get("preferences_md", "data/preferences.md"))
    lifts = compute_lifts(feedback_path)
    apply_lifts(head, lifts)
    apply_manual_hints(head, load_manual_hints(preferences_md))

    # Re-rank within each section after lifts.
    min_rel = int(cfg.get("ranking", {}).get("min_relevance", 3))
    quotas = cfg.get("ranking", {}).get("kind_quotas", {})  # e.g. {twitter: {min: 1}, paper: {max: 1}}

    def _display_sort_key(it: dict):
        return (-int(it.get("relevance_score", 5)), -float(it.get("score", 0)))

    def is_finance(it):  return it.get("category") == "finance"

    max_per_source     = int(cfg.get("ranking", {}).get("max_per_source", 2))
    max_per_subsection = int(cfg.get("ranking", {}).get("max_per_subsection", 3))

    def apply_quotas(section: list[dict], cap: int) -> list[dict]:
        """Greedy fill respecting all caps, then enforce per-kind mins by swap.

        Layered caps so no single feed or topic crowds out the rest:
          - kind_max         (e.g. paper <= 1)
          - kind_min         (e.g. tech blog >= 1)
          - max_per_source   (no single feed contributes >N items)
          - max_per_subsection (each subsection bucket gets <=N items)
        """
        section = [it for it in section if int(it.get("relevance_score", 5)) >= min_rel]
        section.sort(key=lambda x: -x["score"])
        kind_max = {k: int(v.get("max", 999)) for k, v in quotas.items() if "max" in v}
        kind_min = {k: int(v.get("min", 0))   for k, v in quotas.items() if v.get("min", 0)}

        out: list[dict] = []
        kind_used: dict[str, int] = {}
        src_used:  dict[str, int] = {}
        sub_used:  dict[str, int] = {}
        for it in section:
            if len(out) >= cap:
                break
            k    = it.get("source_kind", "tech blog")
            src  = it.get("source", "")
            sub  = it.get("subsection", "")
            if kind_used.get(k, 0)  >= kind_max.get(k, 999):       continue
            if src_used.get(src, 0) >= max_per_source:             continue
            if sub and sub_used.get(sub, 0) >= max_per_subsection: continue
            out.append(it)
            kind_used[k]  = kind_used.get(k, 0) + 1
            src_used[src] = src_used.get(src, 0) + 1
            if sub:
                sub_used[sub] = sub_used.get(sub, 0) + 1
        # Pass 2: enforce min quotas — if a required kind didn't make it, swap
        # the lowest-scoring item from an over-represented kind for the best
        # candidate of the under-represented kind.
        for k, need in kind_min.items():
            have = sum(1 for it in out if it.get("source_kind") == k)
            if have >= need:
                continue
            candidates = [it for it in section if it.get("source_kind") == k and it not in out]
            for c in candidates:
                if sum(1 for it in out if it.get("source_kind") == k) >= need:
                    break
                # find item to evict: lowest-score item NOT itself under its min
                evictable = [it for it in out
                             if sum(1 for x in out if x.get("source_kind") == it.get("source_kind"))
                                > kind_min.get(it.get("source_kind"), 0)]
                if not evictable:
                    if len(out) < cap:
                        out.append(c)
                    continue
                victim = sorted(evictable, key=lambda x: x["score"])[0]
                out.remove(victim)
                out.append(c)
        # Sort what we return by what the user sees (relevance chip), with
        # numeric score as tiebreaker. This keeps the visible order consistent
        # with the chip — no more "7/10 listed below 6/10".
        out.sort(key=_display_sort_key)
        return out

    tech_items    = apply_quotas([it for it in head if not is_finance(it)], tech_top_n)
    finance_items = apply_quotas([it for it in head if is_finance(it)],     finance_top_n)
    items = tech_items + finance_items
    total_ranked = len(head)

    # 7) Two executive summaries — one per persona.
    exec_summary_eng = _build_executive_summary(tech_items,    cfg.get("bedrock", {}), ENGINEER)
    exec_summary_inv = _build_executive_summary(finance_items, cfg.get("bedrock", {}), INVESTOR)

    # 8) Optional: synthesize audio.
    server_cfg = cfg.get("server", {})
    audio_cfg = cfg.get("audio", {})
    # URL resolution order for links rendered in the email:
    #   1) cloudflared tunnel URL (auto-written by `news-digest tunnel` to data/tunnel_url.txt)
    #   2) explicitly configured server.public_base_url
    #   3) localhost base_url (only useful when reading on the same Mac)
    server_url = None
    if server_cfg.get("enabled"):
        from .tunnel import current_tunnel_url
        tunnel_path = server_cfg.get("tunnel_url_path", "data/tunnel_url.txt")
        tunnel_path = Path(tunnel_path)
        if not tunnel_path.is_absolute():
            tunnel_path = ROOT / tunnel_path
        server_url = (current_tunnel_url(tunnel_path)
                      or server_cfg.get("public_base_url")
                      or server_cfg.get("base_url"))
    if audio_cfg.get("enabled") and server_url:
        generate_audio(items, audio_cfg)
    else:
        for it in items:
            it.pop("audio_url", None)

    # 9) Write items_index.json so the server can resolve item_id → metadata.
    if server_url:
        idx_path = Path(cfg["storage"].get("items_index", "data/items_index.json"))
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        idx_path.write_text(json.dumps({it["id"]: it for it in items}, default=str))

    # 10) Render — each section has its own subsections (LLM-assigned) + top-of-mind.
    def group_by_subsection(persona, persona_items):
        groups: dict[str, list[dict]] = {sub: [] for sub in persona.subsections}
        for it in persona_items:
            sub = it.get("subsection") or persona.subsections[-1]
            groups.setdefault(sub, []).append(it)
        # Drop empty subsections, preserve persona's order.
        out = []
        for sub in persona.subsections:
            if groups.get(sub):
                out.append({"label": sub, "entries": groups[sub]})
        # Anything assigned to a non-canonical subsection ends up under last bucket already.
        return out

    sections = [
        {
            "title": ENGINEER.name,
            "executive_summary": exec_summary_eng,
            "subsections": group_by_subsection(ENGINEER, tech_items),
            "entries": tech_items,           # also keep flat list for fallback rendering
        },
        {
            "title": INVESTOR.name,
            "executive_summary": exec_summary_inv,
            "subsections": group_by_subsection(INVESTOR, finance_items),
            "entries": finance_items,
        },
    ]
    html = render_html(sections,
                       failures=[],
                       server_url=server_url,
                       total_ranked=total_ranked)
    text = render_text(sections)

    today_short = datetime.now().strftime("%a · %b %d, %Y")
    subject = f"ML-infra digest — {today_short}"

    if preview:
        out_path = Path(cfg["output"].get("preview_path", "data/preview.html"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html)
        print(f"preview written to {out_path} ({len(items)} items)")
    else:
        recipients = cfg.get("recipients", [])
        if not recipients:
            raise SystemExit("config.recipients is empty")
        mail_send(cfg["smtp"], recipients, subject, html, text)

    # 11) Mark seen (with sent_at if this was a real send), prune cache + dedup,
    #     refresh preferences.md.
    storage.mark_seen(items, sent=not preview)
    storage.prune(int(cfg["storage"].get("retain_days", 60)))
    regenerate_preferences_md(feedback_path, preferences_md)
    try:
        from .http_cache import prune as prune_http
        n = prune_http(max_age_days=int(cfg.get("cache", {}).get("max_age_days", 7)))
        if n: logging.info("pruned %d stale HTTP cache entries", n)
    except Exception:
        pass

    return {"items": len(items)}


# --------------- subcommands ---------------

def cmd_run(args, cfg):       run_pipeline(cfg, preview=False, fetch_only=False)
def cmd_preview(args, cfg):   run_pipeline(cfg, preview=True,  fetch_only=False)
def cmd_fetch_only(args, cfg): run_pipeline(cfg, preview=False, fetch_only=True)


def cmd_discover(args, cfg):
    result = run_discovery()
    print(f"wrote {result['hn_domains']} candidate domains to {result['path']}")
    print("Review and copy useful ones into config.yaml under `sources:`.")


def cmd_serve(args, cfg):
    from . import server
    def _abs(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else ROOT / path
    server.CONFIG["feedback_path"]  = _abs(cfg["storage"].get("feedback_path",  "data/feedback.jsonl"))
    server.CONFIG["preferences_md"] = _abs(cfg["storage"].get("preferences_md", "data/preferences.md"))
    server.CONFIG["audio_dir"]      = _abs(cfg.get("audio", {}).get("output_dir", "data/audio"))
    server.CONFIG["items_index"]    = _abs(cfg["storage"].get("items_index", "data/items_index.json"))
    s = cfg.get("server", {})
    host = s.get("host", "127.0.0.1")
    port = int(s.get("port", 9123))
    server.serve(host, port)


def cmd_tunnel(args, cfg):
    """Long-running cloudflared wrapper used by the tunnel launchd job."""
    from . import tunnel
    s = cfg.get("server", {})
    port = int(s.get("port", 9123))
    path = Path(cfg.get("server", {}).get("tunnel_url_path", "data/tunnel_url.txt"))
    if not path.is_absolute():
        path = ROOT / path
    return tunnel.run_tunnel_forever(port, path)


def cmd_add_recipient(args, cfg):
    rcpts = cfg.setdefault("recipients", [])
    if args.email in rcpts:
        print(f"already present: {args.email}"); return
    rcpts.append(args.email)
    save_config(cfg, Path(args.config) if args.config else None)
    print(f"added: {args.email}")


def cmd_remove_recipient(args, cfg):
    rcpts = cfg.get("recipients", [])
    if args.email not in rcpts:
        print(f"not present: {args.email}"); return
    rcpts.remove(args.email)
    save_config(cfg, Path(args.config) if args.config else None)
    print(f"removed: {args.email}")


def cmd_list_sources(args, cfg):
    for s in cfg.get("sources", []):
        flag = "✓" if s.get("enabled", True) else "✗"
        print(f"  {flag} [{s.get('category','?'):>11}] w={s.get('weight',1.0):.1f}  {s['name']}")
    print(f"\n  Twitter handles: {', '.join(cfg.get('twitter', {}).get('handles', [])) or '(none)'}")
    print(f"  arXiv authors:   {', '.join(a['name'] for a in cfg.get('arxiv_authors', {}).get('authors', [])) or '(none)'}")


def cmd_tune_subsections(args, cfg):
    """Ask the model to propose subsection edits; write data/proposed_subsections.yaml."""
    from .subsection_tuner import propose, write_proposal
    _ensure_aws_sso(cfg)
    proposal = propose(cfg, days=args.days)
    out_path = Path(cfg.get("storage", {}).get("proposed_subsections", "data/proposed_subsections.yaml"))
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    write_proposal(proposal, out_path)
    print(f"Proposal written to {out_path}\n")
    for pkey, plan in proposal.items():
        edits = plan.get("edits", [])
        print(f"=== {pkey} ===")
        print(f"  Rationale: {plan.get('rationale','')}")
        if not edits:
            print("  No edits proposed.\n"); continue
        for e in edits:
            op = e.get("op")
            if op == "add":    print(f"  + ADD '{e.get('label')}' after '{e.get('after')}'")
            elif op == "rename": print(f"  ~ RENAME '{e.get('from')}' → '{e.get('to')}'")
            elif op == "remove": print(f"  − REMOVE '{e.get('label')}' ({e.get('reason','')})")
        print()
    print("Apply with:  news-digest apply-subsections")


def cmd_apply_subsections(args, cfg):
    """Apply the proposal in data/proposed_subsections.yaml to config.yaml."""
    from .subsection_tuner import apply_proposal
    in_path = Path(cfg.get("storage", {}).get("proposed_subsections", "data/proposed_subsections.yaml"))
    if not in_path.is_absolute():
        in_path = ROOT / in_path
    if not in_path.exists():
        raise SystemExit(f"no proposal at {in_path} — run `tune-subsections` first")
    proposal = yaml.safe_load(in_path.read_text())
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG
    applied = apply_proposal(proposal, config_path)
    print("Applied subsection overrides to config.yaml:")
    for pkey, subs in applied.items():
        print(f"  {pkey}: {subs}")


def cmd_history(args, cfg):
    """Print recently-sent items so you can see what's been delivered."""
    s = Store(cfg["storage"]["sqlite_path"])
    items = s.recent_sent(days=args.days)
    if not items:
        print(f"(no items sent in the last {args.days} days)")
        return
    print(f"# Recently-sent items (last {args.days} days, {len(items)} total)\n")
    last_day = None
    for it in items:
        day = it["sent_at"][:10]
        if day != last_day:
            print(f"\n## {day}")
            last_day = day
        print(f"  · {it['source']:<24}  {it['title'][:80]}")


def cmd_test_smtp(args, cfg):
    rcpts = cfg.get("recipients", [])
    if not rcpts:
        raise SystemExit("config.recipients is empty")
    mail_send(cfg["smtp"], rcpts[:1], "news-digest SMTP test",
              "<p>SMTP is working.</p>", "SMTP is working.")
    print(f"sent test to {rcpts[0]}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="news-digest")
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").set_defaults(fn=cmd_run)
    sub.add_parser("preview").set_defaults(fn=cmd_preview)
    sub.add_parser("fetch-only").set_defaults(fn=cmd_fetch_only)
    sub.add_parser("discover").set_defaults(fn=cmd_discover)
    sub.add_parser("serve").set_defaults(fn=cmd_serve)
    sub.add_parser("tunnel").set_defaults(fn=cmd_tunnel)
    sub.add_parser("list-sources").set_defaults(fn=cmd_list_sources)
    sub.add_parser("test-smtp").set_defaults(fn=cmd_test_smtp)
    hist = sub.add_parser("history"); hist.add_argument("--days", type=int, default=14); hist.set_defaults(fn=cmd_history)
    tune = sub.add_parser("tune-subsections"); tune.add_argument("--days", type=int, default=7); tune.set_defaults(fn=cmd_tune_subsections)
    sub.add_parser("apply-subsections").set_defaults(fn=cmd_apply_subsections)
    add = sub.add_parser("add-recipient");    add.add_argument("email"); add.set_defaults(fn=cmd_add_recipient)
    rm  = sub.add_parser("remove-recipient"); rm.add_argument("email");  rm.set_defaults(fn=cmd_remove_recipient)

    args = p.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    _setup_logging(cfg.get("output", {}).get("log_dir", "logs"))
    apply_overrides_from_config(cfg)
    args.fn(args, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
