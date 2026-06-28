# news-digest

A daily curated email — one section for **ML Infra & Tech** (scored by a senior ML-infra engineer persona) and one for **Finance & Markets** (scored by a US tech-investor persona), each with its own top-of-mind synthesis, subsections, and ranked items. Built for a single reader's daily inbox, runs on a laptop, costs about $0.01/run.

```
┌─ Daily Digest — Wednesday, June 25, 2026 ─────────────────────┐
│  20 items ranked from ~38 sources                              │
├────────────────────────────────────────────────────────────────┤
│  ML INFRA & TECH                                               │
│    [Top of mind] one headline + 2-3 themed bullet groups       │
│    -- Models & Training --                                     │
│    -- Inference & Serving --                                   │
│    -- Agents, Tools & Harnesses --                             │
│    -- Research & Papers --                                     │
│    -- Learning & Courses --                                    │
│    -- Industry & Discussion --                                 │
├────────────────────────────────────────────────────────────────┤
│  FINANCE & MARKETS                                             │
│    [Top of mind] one headline + 2-3 themed bullet groups       │
│    -- Macro & Rates --                                         │
│    -- AI Capex & Compute --                                    │
│    -- Semiconductors & Storage --                              │
│    -- Tech Earnings & Strategy --                              │
│    -- Market Sentiment & Flows --                              │
└────────────────────────────────────────────────────────────────┘
```

Each item has: source-kind chip · use-case chip · LLM relevance score · TLDR · 3-5 bullets · why-it-matters bullets · how-to-use bullets · contrarian "take" · clickable "explore next" links · X discussion (when found) · 👍/👎/🔊 action buttons (when the tunnel is up).

## Setup

```bash
git clone https://github.com/<your-account>/ml-news-digest.git
cd ml-news-digest
./scripts/bootstrap.sh
$EDITOR config.yaml                                              # set recipients, schedule
security delete-generic-password -s news-digest-smtp 2>/dev/null
security add-generic-password -a "$USER" -s news-digest-smtp -w 'GMAIL_APP_PW'

# One-time login (Bedrock + cloudflared install if missing)
aws sso login --profile claude
brew install cloudflared                                         # for mobile-readable links

# Verify the pipeline
./scripts/run.sh preview                                         # writes data/preview.html
open data/preview.html
./scripts/run.sh test-smtp

# Install persistent jobs
./scripts/install_launchd.sh                                     # daily digest at config.schedule.send_hour (default 10am local)
./scripts/install_server_launchd.sh                              # Flask server for 👍/👎/🔊 routes
./scripts/install_tunnel_launchd.sh                              # cloudflared public URL for mobile access
```

Uninstall: `scripts/uninstall_*launchd.sh`.

## Configuration knobs

All in `config.yaml`. Most users only touch:

| Section | What it controls |
|---|---|
| `schedule.send_hour` / `send_minute` | When the cron fires (local time) |
| `recipients` | Email addresses (one msg per recipient) |
| `ranking.engineer_top_n` / `investor_top_n` | How many items per section |
| `ranking.kind_quotas` | e.g. `paper: {max: 1}` to suppress paper spam |
| `bedrock.region` | `us-west-2` for Bay Area; `us-east-1` for East coast |
| `bedrock.model_id` | Per-item summaries — Haiku is fast + cheap |
| `bedrock.exec_summary_model_id` | Top-of-mind — Opus 4.7 for sharper synthesis |
| `sources` | Toggle individual feeds; tune `weight` |
| `twitter.handles` | X accounts to track |
| `reddit.subreddits` | r/ML, r/LocalLLaMA, r/wallstreetbets, etc. |
| `audio.provider` | `say` (macOS, free) or `polly` (AWS, $$, needs IAM) |
| `server.public_base_url` | Override for a stable named tunnel |

## Architecture

```
config.yaml
   │
   ├──► fetcher.py        (RSS/Atom + Anthropic HTML scrape, cached on disk)
   ├──► twitter.py        (X API → syndication → RSSHub → Nitter chain)
   ├──► huggingface.py    (HF /api/models trending + /api/daily_papers)
   ├──► arxiv_authors.py  (arXiv Atom API per author)
   ├──► reddit.py         (Reddit RSS, 10s spacing, 429 retries)
   │
   ▼
storage.py (sqlite dedup; `sent_at` timestamp prevents re-sending)
   │
   ▼
personas.py  (engineer scorer + investor scorer, each with its own
              keyword weights, subsection taxonomy, system prompt)
   │
   ▼
summarizer.py  (Bedrock Claude Haiku per item → JSON with tldr / bullets /
                impact_bullets / utilize_bullets / discussion / explore /
                use_case / subsection / relevance_score)
   │
   ▼
preferences.py (lifts from feedback.jsonl + manual BOOST/SUPPRESS hints)
   │           weights: click=0.3, up=1.5, down=-1.5, slope=0.10
   │           caps:    [0.6×, 1.8×] per dimension, [0.4×, 2.2×] per item
   ▼
quotas + final cap per section (kind_quotas: paper max 1, twitter min 1, …)
   │
   ▼
twitter_discuss.py  (DDG → Bing search for `site:x.com <key terms>`,
                     boosts score by hit count, renders tweet links)
   │
   ▼
audio.py  (macOS `say` → AAC/MP3; or AWS Polly when IAM allows)
   │
   ▼
formatter.py  (Jinja2 HTML; one block per section: top-of-mind +
               subsection groups + per-item card)
   │
   ▼
mailer.py  (SMTP STARTTLS to Gmail, password from macOS Keychain)
   │
   ▼
server.py  (Flask, localhost:9123 + cloudflared public URL)
        /click /up /down /audio /listen /listen-all
        appends to feedback.jsonl; preferences.md regenerated each run
```

## Data files

```
data/
  digest.db                # sqlite — seen items, sent timestamps
  feedback.jsonl           # append-only log of clicks/votes
  preferences.md           # human-readable view; supports BOOST/SUPPRESS hints
  items_index.json         # current run's items, for the local server
  preview.html             # last preview output
  tunnel_url.txt           # current public cloudflared URL
  audio/<item_id>.mp3      # per-item TTS
  audio/combined.mp3       # "listen to all"
  http_cache/              # 6h disk cache of fetched feeds
logs/
  digest-YYYY-MM-DD.log    # pipeline run logs
  launchd.stdout.log       # daily cron output
  server.stdout.log        # Flask server
  tunnel.stdout.log        # cloudflared
```

## How feedback shapes future runs

Open the email, click a title (logs a click via `/click/<id>`), tap 👍 or 👎 if a piece really matters. The next day's run:

1. Reads `data/feedback.jsonl` (60-day window).
2. Computes per-source, per-domain, and per-keyword lift multipliers — *dampened*: a single 👍 only adds ~15% to that dimension; the per-item compound is capped at 2.2×, so feedback nudges ranking but doesn't dominate it.
3. Regenerates `data/preferences.md` so you can see what the system thinks you like.
4. You can hand-edit a `BOOST: foo, bar` / `SUPPRESS: x, y` block inside the `<!-- MANUAL_HINTS_*-->` markers — those survive auto-regeneration.

## Twitter strategy

X removed free access. The pipeline tries four paths in order, per handle:

1. **X API v2** if `TWITTER_BEARER_TOKEN` env var is set ($100/mo).
2. **`syndication.twitter.com`** (no-auth embed widget; rate-limits hard per IP).
3. **RSSHub** at `twitter.rsshub.base_url` — self-host: `docker run -d -p 1200:1200 diygod/rsshub`, then set `base_url: http://localhost:1200`.
4. **Nitter pool** with rotation + retries (most public instances 403 in 2026).

If none work, the Twitter section is empty; r/LocalLLaMA + r/MachineLearning fill some of the same gap.

## CLI

```
scripts/run.sh run                   # fetch + summarize + send to all recipients
scripts/run.sh preview               # same, but writes data/preview.html (no SMTP)
scripts/run.sh fetch-only            # count items per source; no LLM, no email
scripts/run.sh serve                 # Flask feedback server
scripts/run.sh tunnel                # cloudflared quick tunnel (writes tunnel_url.txt)
scripts/run.sh discover              # mine HN for new trending domains
scripts/run.sh history --days 14     # show items previously sent
scripts/run.sh list-sources
scripts/run.sh add-recipient EMAIL   # edits config.yaml
scripts/run.sh remove-recipient EMAIL
scripts/run.sh test-smtp             # 1-line email to first recipient
```

## Claude integration

A Claude skill at `~/.claude/skills/news-digest/SKILL.md` lets any Claude Code session operate the pipeline: "run my digest", "what's in today's digest", "add foo@bar.com", "change digest time to 7am", "what did I click last week", etc.

## Cost estimate per run

- Per-item summaries (Haiku, ~20 items × ~1.5k input + 800 output): ~$0.005
- Top-of-mind synthesis (Opus 4.7, 2 calls × ~600 output): ~$0.005
- Audio (macOS `say`): free
- Tunnel + SMTP: free
- **Total: ~$0.01/day** = $3-4/year

## Files

```
src/news_digest/
  main.py             CLI + pipeline orchestration
  personas.py         engineer + investor scorers, subsection taxonomies
  fetcher.py          RSS/Atom + Anthropic scraper (with disk cache)
  twitter.py          X API → syndication → RSSHub → Nitter chain
  twitter_discuss.py  DDG/Bing search for "site:x.com <terms>"
  reddit.py           Reddit RSS with 10s spacing + 429 retries
  huggingface.py      HF /api/models + /api/daily_papers
  arxiv_authors.py    arXiv Atom API per-author queries
  discovery.py        HN-based new-source surfacing
  summarizer.py       Bedrock Haiku; per-item JSON output with retries
  preferences.py      feedback aggregation + lift computation
  audio.py            Polly + macOS `say` TTS
  server.py           Flask: /click /up /down /audio /listen /listen-all
  tunnel.py           cloudflared quick-tunnel runner + URL rotation
  http_cache.py       disk-backed HTTP response cache (6h TTL)
  formatter.py        Jinja2 HTML + plain-text fallback
  mailer.py           SMTP STARTTLS, macOS Keychain for password
  storage.py          SQLite dedup with sent_at column
templates/digest.html.j2
scripts/
  bootstrap.sh
  install_launchd.sh           uninstall_launchd.sh
  install_server_launchd.sh    uninstall_server_launchd.sh
  install_tunnel_launchd.sh    uninstall_tunnel_launchd.sh
  launchd.plist.template
  server.plist.template
  tunnel.plist.template
  run.sh
config.example.yaml
```
