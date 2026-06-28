"""Weekly subsection tuner.

Looks at recently-sent items, asks Opus to propose 0-3 subsection edits per
persona (add a hot new bucket, rename a stale one), then writes the proposal
to `data/proposed_subsections.yaml` for the user to apply via:

    news-digest apply-subsections

The pipeline does NOT auto-mutate the live config — subsections shape the LLM
prompt, and silent changes would shift digest output unexpectedly.
"""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path

import boto3
import yaml

from .personas import PERSONAS, ENGINEER, INVESTOR
from .storage import Store

log = logging.getLogger(__name__)


def _bedrock_client(region: str):
    return boto3.client("bedrock-runtime", region_name=region)


PROMPT = """\
You are tuning the subsection taxonomy of a daily news digest for {role}.

CURRENT subsections (in order they appear):
{current}

Sample of items shipped in the last {days} days under this persona:
{sample}

Propose 0–3 *minimal* edits to the subsection list. Prefer keeping the existing
buckets. Only suggest a new subsection if there is a sustained, high-quality
thread in the sample that doesn't fit existing buckets. Only suggest a rename
if the existing label is genuinely stale.

Output JSON, no preamble:
{{
  "rationale": "<1-2 sentences explaining your edits, citing examples from the sample>",
  "edits": [
    {{"op": "add",    "label": "<new label, 3-5 words>", "after": "<existing label this should follow, or null to append>"}},
    {{"op": "rename", "from":  "<existing label>",       "to":    "<new label, 3-5 words>"}},
    {{"op": "remove", "label": "<existing label>", "reason": "<why retire>"}}
  ]
}}

If no changes are warranted, return {{"rationale": "...", "edits": []}}.
"""


def _propose_for_persona(client, model_id: str, persona, items: list[dict], days: int) -> dict:
    sample_lines = []
    for it in items[:60]:
        sub = it.get("subsection") or ""
        sample_lines.append(f"- [{sub}] {it['title'][:120]}")
    prompt = PROMPT.format(
        role=persona.role_description,
        current="\n".join(f"  {i+1}. {s}" for i, s in enumerate(persona.subsections)),
        sample="\n".join(sample_lines) if sample_lines else "(no items yet)",
        days=days,
    )
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 700,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = client.invoke_model(
        modelId=model_id, body=json.dumps(body),
        contentType="application/json", accept="application/json",
    )
    text = json.loads(resp["body"].read())["content"][0]["text"].strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().startswith("json"): text = text.lstrip()[4:]
    try:
        return json.loads(text.strip())
    except Exception as e:
        log.warning("subsection tuner failed to parse JSON for %s: %s", persona.key, e)
        return {"rationale": "(parse failed)", "edits": []}


def propose(cfg: dict, days: int = 7) -> dict:
    """Return {persona_key: {rationale, edits, current_subsections}}."""
    store = Store(cfg["storage"]["sqlite_path"])
    history = store.recent_sent(days=days)
    if not history:
        return {p.key: {"rationale": "(no history yet)", "edits": []} for p in PERSONAS}

    # We don't store category in history; infer from source heuristically.
    # For now use ALL items for both personas; the prompt makes each persona
    # see only items that match its taxonomy concerns.
    bedrock_cfg = cfg.get("bedrock", {})
    model_id = bedrock_cfg.get("exec_summary_model_id") or bedrock_cfg["model_id"]
    client = _bedrock_client(bedrock_cfg.get("region", os.environ.get("AWS_REGION", "us-east-1")))

    FINANCE_SOURCES = {
        "WSJ Markets","FT Alphaville","Stratechery","Marginal Revolution",
        "Noahpinion","Matt Stoller BIG","r/wallstreetbets","r/stocks","r/investing",
    }
    out: dict = {}
    for p in PERSONAS:
        if p.key == "investor":
            items_for_p = [h for h in history if h["source"] in FINANCE_SOURCES]
        else:
            items_for_p = [h for h in history if h["source"] not in FINANCE_SOURCES]
        proposal = _propose_for_persona(client, model_id, p, items_for_p, days)
        proposal["current_subsections"] = list(p.subsections)
        out[p.key] = proposal
    return out


def write_proposal(proposal: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(proposal, sort_keys=False, allow_unicode=True))


def apply_proposal(proposal: dict, config_path: Path) -> dict:
    """Apply the edits to config.yaml's `personas.subsection_overrides` block.

    We don't mutate src/news_digest/personas.py; instead the runtime merges
    `personas.subsection_overrides.<persona_key>` from config.yaml on top of
    the defaults at start-up.
    """
    cfg = yaml.safe_load(config_path.read_text())
    cfg.setdefault("personas", {}).setdefault("subsection_overrides", {})
    applied: dict[str, list[str]] = {}
    for pkey, plan in proposal.items():
        # Find the persona's current ordered subsection list (default + previous overrides).
        persona = ENGINEER if pkey == "engineer" else INVESTOR
        current = list(cfg["personas"]["subsection_overrides"].get(pkey) or persona.subsections)
        for edit in plan.get("edits", []):
            op = edit.get("op")
            if op == "add":
                label = edit.get("label")
                if not label or label in current:
                    continue
                after = edit.get("after")
                if after and after in current:
                    idx = current.index(after) + 1
                    current.insert(idx, label)
                else:
                    current.append(label)
            elif op == "rename":
                src, dst = edit.get("from"), edit.get("to")
                if src in current and dst:
                    current[current.index(src)] = dst
            elif op == "remove":
                if edit.get("label") in current:
                    current.remove(edit["label"])
        cfg["personas"]["subsection_overrides"][pkey] = current
        applied[pkey] = current
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return applied
