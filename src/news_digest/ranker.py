"""Keyword + recency + source-weight ranker. No LLM needed.

The summarizer is the expensive step, so we rank first and only summarize the top N
per section. Score = source_weight * (keyword_hits * w_kw + recency_bonus + length_penalty).
"""
from __future__ import annotations
from datetime import datetime, timezone
import math
import re

# Tuned for the ML-infra-SWE persona. Order matters — longer phrases tried first.
ML_INFRA_KEYWORDS = {
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
    "vllm": 3.0, "sglang": 3.0, "tensorrt": 2.5, "tensorrt-llm": 3.0,
    "tgi": 2.0, "triton": 2.5, "kv cache": 3.0, "kv-cache": 3.0,
    "paged attention": 3.0, "flash attention": 3.0, "flashattention": 3.0,
    "speculative decod": 3.0, "continuous batch": 3.0, "prefill": 2.5,
    "kernel": 1.5, "fused": 1.5, "torch.compile": 2.5, "inductor": 2.0,
    "xla": 2.0, "mlir": 2.0,
    # Models / arch
    "moe": 2.5, "mixture-of-experts": 2.5, "mixture of experts": 2.5,
    "long context": 2.0, "rope": 1.5, "ssm": 1.8, "mamba": 1.8,
    # Post-training / RL
    "rlhf": 2.0, "dpo": 2.0, "grpo": 2.5, "ppo": 1.5, "reward model": 1.8,
    "rl ": 1.5, "post-train": 2.0,
    # Production / cost
    "throughput": 1.5, "latency": 1.5, "tokens/sec": 2.0, "tok/s": 2.0,
    "cost per token": 2.5, "$/token": 2.5, "capex": 1.2,
    # Frameworks
    "pytorch": 1.2, "jax": 1.5, "tensorflow": 0.8,
}

FINANCE_KEYWORDS = {
    "fed": 1.5, "rates": 1.0, "inflation": 1.0, "treasur": 1.2,
    "nvidia": 1.5, "tsmc": 1.5, "asml": 1.2, "semiconductor": 1.2,
    "capex": 1.5, "ai capex": 2.5, "data center": 1.5, "datacenter": 1.5,
    "ipo": 1.0, "earnings": 0.8, "merger": 0.8, "antitrust": 1.0,
    "energy": 1.0, "power grid": 1.5, "nuclear": 1.2,
}

PAPER_KEYWORDS = {
    "scaling law": 2.5, "ablation": 1.0, "benchmark": 1.0, "sota": 1.5,
    "transformer": 1.0, "attention": 1.2, "diffusion": 1.5, "rlhf": 1.5,
}

NOISE_KEYWORDS = {
    # Down-weight political/celebrity/clickbait that sneaks in via aggregators.
    "trump": -2.0, "biden": -2.0, "election": -1.5, "celebrit": -2.0,
    "show hn:": -0.5, "ask hn:": -0.5,
}


def _kw_score(text: str, kws: dict[str, float]) -> float:
    if not text:
        return 0.0
    t = text.lower()
    s = 0.0
    for k, w in kws.items():
        if k in t:
            s += w
    return s


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
    # Decay over 48h: fresh items get +2.0, day-old +1.0, two-day-old ~0.
    return max(0.0, 2.0 * math.exp(-hours / 24.0))


def score_item(item: dict) -> float:
    text = f"{item.get('title', '')} . {item.get('summary', '')}"
    cat = item.get("category", "ml_infra")
    if cat == "finance":
        base = _kw_score(text, FINANCE_KEYWORDS)
    elif cat == "papers":
        base = _kw_score(text, PAPER_KEYWORDS) + _kw_score(text, ML_INFRA_KEYWORDS) * 0.5
    elif cat == "twitter":
        base = _kw_score(text, ML_INFRA_KEYWORDS) + _kw_score(text, FINANCE_KEYWORDS) * 0.3
    else:
        base = _kw_score(text, ML_INFRA_KEYWORDS)
    base += _kw_score(text, NOISE_KEYWORDS)
    base += _recency_bonus(item)
    base *= float(item.get("weight", 1.0))
    return base


def score_items(items: list[dict]) -> list[dict]:
    """Score items in place and return them sorted high→low."""
    for it in items:
        it["score"] = score_item(it)
    return sorted(items, key=lambda x: -x["score"])
