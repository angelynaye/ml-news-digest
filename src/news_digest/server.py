"""Local Flask server for click tracking, votes, and audio playback.

Endpoints:
  GET /click/{item_id}?u=<url>     log a click and 302 to <url>
  GET /up/{item_id}                log an upvote, show a confirmation page
  GET /down/{item_id}              log a downvote, show a confirmation page
  GET /audio/{item_id}.mp3         stream the per-item MP3
  GET /audio/combined.mp3          stream the combined MP3
  GET /listen/{item_id}            HTML page with an <audio> player for the item
  GET /listen-all                  HTML page with the combined player + item list

Run: scripts/run.sh serve  (or via launchd plist installed by install_server.sh)
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, redirect, request, send_file, abort, Response

from .preferences import append_event, regenerate_preferences_md

log = logging.getLogger(__name__)
app = Flask(__name__)

# Set at startup by main.serve_cli; default for local dev.
CONFIG = {
    "feedback_path": Path("data/feedback.jsonl"),
    "preferences_md": Path("data/preferences.md"),
    "audio_dir": Path("data/audio"),
    "items_index": Path("data/items_index.json"),
}


def _load_item(item_id: str) -> dict | None:
    p = CONFIG["items_index"]
    if not p.exists():
        return None
    try:
        idx = json.loads(p.read_text())
        return idx.get(item_id)
    except Exception:
        return None


@app.route("/click/<item_id>")
def click(item_id):
    url = request.args.get("u", "")
    item = _load_item(item_id) or {}
    append_event(CONFIG["feedback_path"], {
        "kind": "click",
        "item_id": item_id,
        "url": url or item.get("url", ""),
        "source": item.get("source", ""),
        "title": item.get("title", ""),
    })
    if not url:
        return abort(400, "missing ?u=")
    return redirect(url, code=302)


def _vote(item_id: str, kind: str) -> Response:
    item = _load_item(item_id) or {}
    append_event(CONFIG["feedback_path"], {
        "kind": kind,
        "item_id": item_id,
        "url": item.get("url", ""),
        "source": item.get("source", ""),
        "title": item.get("title", ""),
    })
    # Re-derive preferences.md so the user can inspect what the system learned.
    try:
        regenerate_preferences_md(CONFIG["feedback_path"], CONFIG["preferences_md"])
    except Exception as e:
        log.warning("regen preferences.md failed: %s", e)
    word = "👍 Up-voted" if kind == "up" else "👎 Down-voted"
    title = item.get("title", item_id)
    return Response(
        f"""<!doctype html><html><body style='font-family:-apple-system;padding:40px;max-width:600px;margin:auto;'>
        <h2>{word}</h2>
        <p style='color:#444'>{title}</p>
        <p style='color:#888;font-size:13px'>Recorded. The next digest will weight similar items accordingly.</p>
        <p><a href='/'>← back</a></p>
        </body></html>""",
        mimetype="text/html",
    )


@app.route("/up/<item_id>")
def up(item_id):
    return _vote(item_id, "up")


@app.route("/down/<item_id>")
def down(item_id):
    return _vote(item_id, "down")


@app.route("/audio/<path:fname>")
def audio(fname):
    p = CONFIG["audio_dir"] / fname
    if not p.exists() or not p.is_file():
        return abort(404)
    return send_file(p, mimetype="audio/mpeg", as_attachment=False)


@app.route("/listen/<item_id>")
def listen_item(item_id):
    item = _load_item(item_id) or {}
    return Response(
        f"""<!doctype html><html><body style='font-family:-apple-system;padding:40px;max-width:680px;margin:auto;'>
        <h2 style='margin-bottom:4px'>{item.get('title', item_id)}</h2>
        <div style='color:#888;font-size:13px;margin-bottom:16px'>{item.get('source', '')}</div>
        <audio controls autoplay style='width:100%'><source src='/audio/{item_id}.mp3'></audio>
        <p style='margin-top:18px'><a href='{item.get('url', '#')}' target='_blank'>Open original article →</a></p>
        </body></html>""", mimetype="text/html")


@app.route("/listen-all")
def listen_all():
    idx_path = CONFIG["items_index"]
    items: list[dict] = []
    if idx_path.exists():
        try:
            items = list(json.loads(idx_path.read_text()).values())
        except Exception:
            pass
    items_html = "".join(
        f"<li><a href='/listen/{it['id']}'>{it['title']}</a> <span style='color:#888'>· {it['source']}</span></li>"
        for it in items
    )
    return Response(
        f"""<!doctype html><html><body style='font-family:-apple-system;padding:40px;max-width:680px;margin:auto;'>
        <h2>Daily digest — combined audio</h2>
        <audio controls autoplay style='width:100%'><source src='/audio/combined.mp3'></audio>
        <h3 style='margin-top:32px'>Individual items</h3>
        <ul>{items_html}</ul>
        </body></html>""", mimetype="text/html")


@app.route("/")
def root():
    return Response("<p>news-digest server. <a href='/listen-all'>Listen to today's digest →</a></p>",
                    mimetype="text/html")


def serve(host: str, port: int) -> None:
    log.info("starting news-digest server on http://%s:%d", host, port)
    app.run(host=host, port=port, debug=False, use_reloader=False)
