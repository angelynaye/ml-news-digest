"""Two persona scorers — engineer & investor — with their own keyword sets,
subsection taxonomies, and LLM prompt fragments.

Items are scored under whichever persona's category bucket they fall into:
  - category in (ml_infra, papers, discussions, twitter) → engineer persona
  - category == finance                                   → investor persona

This file is the single source of truth for "what does each persona care about".
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from datetime import datetime, timezone


@dataclass
class Persona:
    key: str                       # short id used in config / sections
    name: str                      # display name for the digest
    role_description: str          # for LLM system prompt
    subsections: list[str]         # the ordered list the LLM picks from
    keywords: dict[str, float]     # keyword → weight; matched against title+summary
    noise_keywords: dict[str, float] = field(default_factory=dict)  # penalize


# ====================================================================
# Engineer persona — senior ML-infra SWE
# ====================================================================

ENGINEER_SUBSECTIONS = [
    "Models & Training",
    "Inference & Serving",
    "Agents, Tools & Harnesses",
    "Research & Papers",
    "Learning & Courses",
    "Industry & Discussion",
]

ENGINEER_KEYWORDS: dict[str, float] = {
    # Hardware
    "h100": 3.0, "h200": 3.0, "b200": 3.5, "blackwell": 3.0, "hopper": 2.5,
    "mi300": 2.5, "tpu": 2.0, "gpu": 1.0, "cuda": 2.0, "nccl": 3.0,
    "rdma": 2.5, "nvlink": 2.5, "infiniband": 2.5,
    # Training systems
    "fsdp": 3.0, "deepspeed": 2.5, "megatron": 3.0, "zero-3": 2.5,
    "pipeline parallel": 3.0, "tensor parallel": 3.0, "data parallel": 1.5,
    "context parallel": 3.0, "sequence parallel": 2.5,
    "mixed precision": 1.8, "bfloat16": 1.8, "fp8": 2.5, "fp4": 2.5,
    "checkpoint": 1.5, "fault toleran": 2.5,
    # Inference / serving
    "vllm": 3.5, "sglang": 3.5, "tensorrt": 2.5, "tensorrt-llm": 3.0,
    "tgi": 2.0, "triton": 2.5, "kv cache": 3.0, "kv-cache": 3.0,
    "paged attention": 3.0, "flash attention": 3.0, "flashattention": 3.0,
    "speculative decod": 3.0, "continuous batch": 3.0, "prefill": 2.5,
    "kernel": 1.5, "fused": 1.5, "torch.compile": 2.5, "inductor": 2.0,
    "xla": 2.0, "mlir": 2.0,
    # Models / architectures
    "moe": 2.5, "mixture-of-experts": 2.5, "mixture of experts": 2.5,
    "long context": 2.0, "rope": 1.5, "ssm": 1.8, "mamba": 1.8,
    # Post-training / RL
    "rlhf": 2.0, "dpo": 2.0, "grpo": 2.5, "ppo": 1.5, "reward model": 1.8,
    "rl ": 1.5, "post-train": 2.0,
    # Agents, tools, harnesses — newly elevated
    "agent": 2.5, "agents": 2.5, "agentic": 2.5,
    "tool use": 2.5, "tool calling": 2.5, "function calling": 2.0,
    "harness": 2.5, "scaffold": 2.0, "loop engineering": 3.0,
    "mcp": 2.5, "model context protocol": 3.0,
    "claude code": 2.5, "cursor": 2.0, "aider": 2.0, "codex": 2.0,
    "computer use": 2.5, "browser agent": 2.5,
    "swe-bench": 2.5, "agentbench": 2.0, "gaia": 1.8,
    # Production / cost
    "throughput": 1.5, "latency": 1.5, "tokens/sec": 2.0, "tok/s": 2.0,
    "cost per token": 2.5, "$/token": 2.5,
    # Frameworks
    "pytorch": 1.2, "jax": 1.5, "tensorflow": 0.8,
    # Learning resources
    "course": 1.8, "tutorial": 1.5, "lecture": 1.5, "video": 1.0, "workshop": 1.2,
    "from scratch": 2.0, "deep dive": 1.5,
}

ENGINEER_NOISE: dict[str, float] = {
    "trump": -2.0, "biden": -1.5, "election": -1.0, "celebrit": -2.0,
    "show hn:": -0.3,
}


# ====================================================================
# Investor persona — US tech investor, semis/AI/storage focus
# ====================================================================

INVESTOR_SUBSECTIONS = [
    "Macro & Rates",
    "AI Capex & Compute",
    "Semiconductors & Storage",
    "Tech Earnings & Strategy",
    "Market Sentiment & Flows",
]

INVESTOR_KEYWORDS: dict[str, float] = {
    # Macro
    "fed": 2.0, "fomc": 2.5, "rate cut": 2.5, "rate hike": 2.5, "rates": 1.2,
    "yield curve": 1.8, "treasur": 1.2, "inflation": 1.5, "cpi": 2.0, "pce": 2.0,
    "jobs report": 1.8, "nonfarm": 1.5, "recession": 1.5, "gdp": 1.2,
    # Semis & storage (the user explicitly called out MU + storage)
    "nvidia": 2.5, "nvda": 2.5, "tsmc": 2.0, "asml": 1.8, "amd": 2.0,
    "intel": 1.5, "broadcom": 1.5, "qualcomm": 1.3,
    "micron": 3.0, "mu ": 2.5, "western digital": 2.5, "wdc": 2.0,
    "seagate": 2.0, "skhynix": 2.5, "sk hynix": 2.5, "samsung": 1.5,
    "hbm": 3.0, "hbm3": 3.0, "hbm4": 3.5, "ddr5": 2.0, "ssd": 1.8,
    "storage": 2.0, "memory": 1.8, "nand": 2.0, "dram": 2.5,
    "semiconductor": 1.8, "foundry": 1.8, "fab ": 1.5, "wafer": 1.5,
    # AI capex / data centers
    "capex": 2.5, "ai capex": 3.5, "data center": 2.5, "datacenter": 2.5,
    "hyperscaler": 2.5, "compute spend": 2.5, "power grid": 2.0,
    "nuclear": 1.8, "ppa": 1.5, "energy demand": 1.8, "cooling": 1.5,
    # Tech companies
    "microsoft": 1.5, "msft": 1.5, "google": 1.3, "googl": 1.3, "alphabet": 1.3,
    "meta": 1.5, "amazon": 1.3, "aws": 1.5, "oracle": 1.8, "orcl": 1.8,
    "apple": 1.2, "tesla": 1.2, "anthropic": 1.5, "openai": 1.5,
    "coreweave": 2.0, "crwv": 2.0, "snowflake": 1.5, "databricks": 1.5,
    # Market action
    "earnings": 1.5, "guidance": 2.0, "beat": 1.2, "miss": 1.5, "raise": 1.2,
    "ipo": 1.8, "spac": 1.0, "merger": 1.2, "acquisition": 1.3, "buyback": 1.2,
    "antitrust": 1.5, "doj": 1.3, "ftc": 1.2,
    # Sentiment & flow
    "options flow": 2.0, "short interest": 1.8, "insider": 1.5,
    "etf flow": 1.5, "hedge fund": 1.3, "13f": 1.2, "wallstreetbets": 1.5,
    "wsb": 1.5, "meme stock": 1.5,
}

INVESTOR_NOISE: dict[str, float] = {
    "celebrit": -2.0, "horoscop": -3.0, "royal family": -2.0,
}


# ====================================================================
# Registry
# ====================================================================

ENGINEER = Persona(
    key="engineer",
    name="ML Infra & Tech",
    role_description=(
        "a senior software engineer working on machine learning infrastructure. "
        "Cares about: distributed training, LLM inference and serving, accelerator "
        "hardware, compilers, agent harnesses, tool use, loop engineering, "
        "courses/videos for advanced ML, and how to deploy AI in production reliably."
    ),
    subsections=ENGINEER_SUBSECTIONS,
    keywords=ENGINEER_KEYWORDS,
    noise_keywords=ENGINEER_NOISE,
)

INVESTOR = Persona(
    key="investor",
    name="Finance & Markets",
    role_description=(
        "a US-focused tech investor. Cares about: macro and rates, AI capex and "
        "semiconductor cycles (Nvidia, TSMC, ASML, AMD), storage and memory (Micron, "
        "WDC, SK Hynix, HBM, NAND/DRAM), hyperscaler infrastructure spending, tech "
        "company earnings and strategy, and market sentiment signals that hint at "
        "what's becoming popular next in tech."
    ),
    subsections=INVESTOR_SUBSECTIONS,
    keywords=INVESTOR_KEYWORDS,
    noise_keywords=INVESTOR_NOISE,
)

PERSONAS = [ENGINEER, INVESTOR]


def persona_for_category(category: str) -> Persona:
    """Map an item's `category` field to a persona."""
    if category == "finance":
        return INVESTOR
    return ENGINEER


def apply_overrides_from_config(cfg: dict) -> None:
    """Merge per-persona subsection lists from config.yaml's
    `personas.subsection_overrides` into the live Persona objects. Called once at
    pipeline startup."""
    overrides = (cfg.get("personas") or {}).get("subsection_overrides") or {}
    for p in PERSONAS:
        if p.key in overrides and isinstance(overrides[p.key], list) and overrides[p.key]:
            p.subsections = list(overrides[p.key])


# ====================================================================
# Scoring
# ====================================================================

def _kw_score(text: str, kws: dict[str, float]) -> float:
    if not text:
        return 0.0
    t = text.lower()
    return sum(w for k, w in kws.items() if k in t)


def _recency_bonus(item: dict) -> float:
    pub = item.get("published_at")
    if not pub:
        return 0.0
    try:
        dt = datetime.fromisoformat(pub)
    except Exception:
        return 0.0
    hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
    if hours < 0:
        return 0.0
    return max(0.0, 2.0 * math.exp(-hours / 24.0))


def score_with_persona(item: dict, persona: Persona) -> float:
    text = f"{item.get('title','')} . {item.get('summary','')}"
    base  = _kw_score(text, persona.keywords)
    base += _kw_score(text, persona.noise_keywords)
    base += _recency_bonus(item)
    base *= float(item.get("weight", 1.0))
    return base


def score_items_by_persona(items: list[dict]) -> dict[str, list[dict]]:
    """Partition + score. Returns {persona_key: [items sorted high→low]}."""
    out: dict[str, list[dict]] = {p.key: [] for p in PERSONAS}
    for it in items:
        persona = persona_for_category(it.get("category", "ml_infra"))
        it["persona"] = persona.key
        it["score"]   = score_with_persona(it, persona)
        out[persona.key].append(it)
    for k in out:
        out[k].sort(key=lambda x: -x["score"])
    return out
