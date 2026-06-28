"""Render the digest HTML + plain-text fallback. Sections each carry their own
executive summary and subsection grouping."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"


def render_html(sections: list[dict], *,
                failures: list[str],
                server_url: str | None,
                total_ranked: int) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    tpl = env.get_template("digest.html.j2")
    now = datetime.now()
    date_long  = now.strftime("%A, %B %d, %Y")
    date_short = now.strftime("%a %b %d")
    total = sum(len(s["entries"]) for s in sections)
    source_set = {it["source"] for s in sections for it in s["entries"]}
    return tpl.render(
        subject=f"Daily digest — {date_short}",
        date_long=date_long,
        date_short=date_short,
        sections=sections,
        failures=failures,
        server_url=server_url,
        counts={
            "total": total,
            "sources": len(source_set),
            "dropped": max(0, total_ranked - total),
        },
        generated_at=now.strftime("%Y-%m-%d %H:%M"),
    )


def render_text(sections: list[dict]) -> str:
    lines: list[str] = [f"Daily digest — {datetime.now().strftime('%A, %B %d, %Y')}", ""]
    for s in sections:
        if not s["entries"]:
            continue
        lines.append("")
        lines.append("=" * 70)
        lines.append(s["title"].upper())
        lines.append("=" * 70)
        es = s.get("executive_summary") or {}
        if es.get("headline"):
            lines.append(f"\nTop of mind: {es['headline']}")
            for theme in es.get("themes", []):
                lines.append(f"\n  {theme['label']}:")
                for b in theme.get("bullets", []):
                    lines.append(f"    • {b}")
            lines.append("")
        for sub in s.get("subsections", []):
            lines.append(f"\n-- {sub['label']} --")
            for i, it in enumerate(sub["entries"], 1):
                lines.append(f"\n{i}. {it['title']}  [{it['source']}]  ({it.get('relevance_score', '-')}/10)")
                if it.get("tldr"):
                    lines.append(f"   {it['tldr']}")
                for b in it.get("bullets", []):
                    lines.append(f"     • {b}")
                if it.get("impact_bullets"):
                    lines.append(f"   Why it matters:")
                    for b in it["impact_bullets"]:
                        lines.append(f"     • {b}")
                if it.get("utilize_bullets"):
                    lines.append(f"   How to use:")
                    for b in it["utilize_bullets"]:
                        lines.append(f"     • {b}")
                if it.get("discussion"):
                    lines.append(f"   Take: {it['discussion']}")
                if it.get("explore"):
                    explore_strs = [(e.get('label','') + f" ({e.get('url','')})") for e in it["explore"]]
                    lines.append(f"   Explore next: " + " | ".join(explore_strs))
                if it.get("x_discussion_urls"):
                    lines.append(f"   X discussion ({it.get('x_discussion_count', 0)}): " +
                                 ", ".join(it["x_discussion_urls"]))
                lines.append(f"   {it['url']}")
    return "\n".join(lines)
