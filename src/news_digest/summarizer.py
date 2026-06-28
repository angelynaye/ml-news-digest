"""Summarize each item richly using Bedrock Claude.

Each item gets: tldr, 3-5 bullets, why-it-matters (impact), what-people-say
(discussion), explore-further suggestions, how-to-utilize.

Falls back to the raw entry summary if Bedrock isn't reachable.
"""
from __future__ import annotations
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

log = logging.getLogger(__name__)

SYSTEM = (
    "You are an editor curating a daily digest. Be terse, technical, no hype, no emoji. "
    "Be concrete; name specific systems, papers, hardware, companies, or numbers when you can."
)


def _is_chinese_item(item: dict) -> bool:
    """True if the item's title or excerpt is predominantly Chinese.

    Heuristic: >= 15% CJK characters in title+summary, OR the source name
    contains explicit Chinese markers.
    """
    text = (item.get("title", "") or "") + " " + (item.get("summary", "") or "")[:300]
    if not text:
        return False
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    if cjk == 0:
        return False
    return cjk / max(1, len(text)) >= 0.15


_CHINESE_INSTRUCTION = (
    "\n\nThis item is in Chinese (中文). Respond in Chinese for `tldr`, `bullets`, "
    "`impact_bullets`, `utilize_bullets`, and `discussion`. KEEP technical terms in "
    "English when they're proper nouns or established jargon (e.g. transformer, "
    "attention, KV cache, RoPE, FlashAttention, vLLM, TPU, GPU, MoE, RLHF, FSDP, "
    "tensor parallel, throughput, latency, fp8, model names like GPT-4 / Claude / "
    "Gemini / Llama). Keep `use_case`, `subsection`, and the `explore[].label` "
    "values in English so they sort consistently with the rest of the digest."
)

PROMPT_TEMPLATE = """ITEM
Title: {title}
Source: {source}
URL: {url}
Excerpt:
{excerpt}

READER: {role_description}

Pick subsection from: {subsections}

Write JSON with exactly these keys:
{{
  "tldr": "1 sentence, <= 35 words. What the item literally is or claims.",
  "bullets": ["3 to 5 short bullets, each <= 22 words. Concrete facts, numbers, system names, the key technical claim or change."],
  "impact_bullets": ["2 to 3 short bullets, each <= 18 words. Concrete impact on ML-infra teams: what unblocks, what's cheaper/faster, what's the catch."],
  "utilize_bullets": ["2 to 3 short bullets, each <= 18 words. Concrete actions: what to prototype, benchmark, swap in, or watch for."],
  "discussion": "1 sentence, <= 30 words. What sharp readers are likely debating or criticizing. If unknown, write 'No public discussion surfaced yet.' — do not invent.",
  "explore": [
    {{"label": "<=10 words, the topic or concept", "url": "<full https URL to a real primary source: arxiv abs page, github repo, paper, docs page. Use https://arxiv.org/abs/<id> for papers, https://github.com/<org>/<repo> for projects, or https://www.google.com/search?q=<URL-encoded query> if no canonical source exists. NEVER fabricate URLs — when uncertain, use the Google search fallback>"}}
  ],
  "use_case": "<one of: announcement | release | tutorial | research | benchmark | discussion | opinion | survey | incident | tooling — pick the single best fit>",
  "subsection": "<one of the subsections listed above — pick the single best fit>",
  "relevance_score": <integer 0-10, how relevant to THIS reader; 0=skip, 10=must-read>
}}
Return ONLY the JSON, no preamble, no markdown fences. Provide 2-3 items in `explore`.
"""


def _bedrock_client(region: str):
    return boto3.client("bedrock-runtime", region_name=region)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    return text.strip()


def _try_parse_json(text: str) -> dict | None:
    """Best-effort: strip fences, then try to repair truncated/malformed JSON."""
    text = _strip_fences(text)
    try:
        return json.loads(text)
    except Exception:
        pass
    # Repair: trim to the last balanced } and try again.
    depth = 0; last_good = -1; in_str = False; esc = False
    for i, ch in enumerate(text):
        if esc:
            esc = False; continue
        if ch == "\\" and in_str:
            esc = True; continue
        if ch == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_good = i
    if last_good > 0:
        try:
            return json.loads(text[: last_good + 1])
        except Exception:
            pass
    return None


def _summarize_one(client, model_id: str, max_tokens: int, item: dict, persona) -> dict:
    excerpt = (item.get("summary") or "")[:1500] or "(no excerpt available)"
    prompt = PROMPT_TEMPLATE.format(
        title=item["title"], source=item["source"],
        url=item["url"], excerpt=excerpt,
        role_description=persona.role_description,
        subsections=" | ".join(persona.subsections),
    )
    if _is_chinese_item(item):
        prompt += _CHINESE_INSTRUCTION
        item["lang"] = "zh"
    else:
        item["lang"] = "en"
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        resp = client.invoke_model(
            modelId=model_id, body=json.dumps(body),
            contentType="application/json", accept="application/json",
        )
        payload = json.loads(resp["body"].read())
        text = payload["content"][0]["text"]
        data = _try_parse_json(text)
        if data is None:
            raise ValueError("could not parse JSON from model output")
        item["tldr"]            = (data.get("tldr") or "").strip()
        item["bullets"]         = [b.strip() for b in (data.get("bullets") or []) if b][:5]
        item["impact_bullets"]  = [b.strip() for b in (data.get("impact_bullets") or []) if b][:3]
        item["utilize_bullets"] = [b.strip() for b in (data.get("utilize_bullets") or []) if b][:3]
        item["discussion"]      = (data.get("discussion") or "").strip()
        # `explore` may be either old-format (list of strings) or new-format (list of {label, url})
        raw_explore = data.get("explore") or []
        explore: list[dict] = []
        for e in raw_explore[:3]:
            if isinstance(e, dict) and e.get("label"):
                label = str(e["label"]).strip()
                url = str(e.get("url", "")).strip()
                if not url.startswith("http"):
                    import urllib.parse as _up
                    url = f"https://www.google.com/search?q={_up.quote_plus(label)}"
                explore.append({"label": label, "url": url})
            elif isinstance(e, str) and e.strip():
                import urllib.parse as _up
                label = e.strip()
                explore.append({
                    "label": label,
                    "url": f"https://www.google.com/search?q={_up.quote_plus(label)}",
                })
        item["explore"]         = explore
        item["use_case"]        = (data.get("use_case") or "").strip().lower() or "discussion"
        item["subsection"]      = (data.get("subsection") or "").strip()
        # Snap-to-canonical so a stray casing/wording variation still groups correctly.
        if item["subsection"]:
            for canon in persona.subsections:
                if canon.lower() == item["subsection"].lower():
                    item["subsection"] = canon
                    break
            else:
                item["subsection"] = persona.subsections[-1]  # fallback bucket
        else:
            item["subsection"] = persona.subsections[-1]
        item["relevance_score"] = int(data.get("relevance_score") or 5)
    except Exception as e:
        log.warning("summarize failed for %s: %s", item.get("url"), e)
        item["tldr"]    = (item.get("summary") or item["title"])[:240]
        item["bullets"] = []
        item["impact_bullets"]  = []
        item["utilize_bullets"] = []
        item["discussion"] = ""
        item["explore"] = []
        item["use_case"] = "discussion"
        item["subsection"] = persona.subsections[-1]
        item["relevance_score"] = 5
    return item


def summarize_items(items: list[dict], bedrock_cfg: dict, persona) -> list[dict]:
    """Summarize a batch of items under one persona's lens.

    If >50% of items fall back (no `bullets` produced), log a loud error so the
    user knows the email is degraded rather than silently shipping empty content.
    Most common cause: expired AWS SSO token.
    """
    if not items:
        return items
    client = _bedrock_client(bedrock_cfg.get("region", os.environ.get("AWS_REGION", "us-east-1")))
    model_id = bedrock_cfg["model_id"]
    max_tokens = int(bedrock_cfg.get("max_tokens", 1100))

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_summarize_one, client, model_id, max_tokens, it, persona): it for it in items}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                log.warning("summarizer worker failed: %s", e)

    # Surface degraded runs prominently.
    failed = sum(1 for it in items if not it.get("bullets"))
    if failed > 0:
        rate = failed / len(items)
        msg = (f"summarizer: {failed}/{len(items)} items got fallback output (no bullets). "
               "Check AWS SSO (`aws sso login --profile claude`) and Bedrock access in the configured region.")
        if rate >= 0.5:
            log.error(msg)
        else:
            log.warning(msg)
    return items
