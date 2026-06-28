"""Manage a Cloudflare quick tunnel for the local feedback server.

Two responsibilities:
  - `current_tunnel_url(path)` — read the latest published URL.
  - `run_tunnel_forever(...)` — long-running entry point used by the launchd
    job. Spawns cloudflared, parses the URL from its output, and *periodically
    health-checks* the URL. Cloudflare's free "quick tunnels" can be reaped
    by their edge while the local cloudflared process keeps running — in that
    case we kill cloudflared so launchd's KeepAlive restarts us with a new URL.

The pipeline doesn't spawn cloudflared itself; one persistent job keeps the
URL fresh so emails always pick up the latest.
"""
from __future__ import annotations
import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
HEALTH_INTERVAL_SEC = 300        # check every 5 minutes
HEALTH_TIMEOUT_SEC = 15
HEALTH_FAILS_BEFORE_RESTART = 2  # require 2 consecutive failures to avoid flaps


def current_tunnel_url(tunnel_url_path: Path, *, probe: bool = True) -> str | None:
    """Read the most recent tunnel URL written by the launchd job.

    If `probe` is True (default), the URL is HEAD-checked before being returned.
    A dead URL — Cloudflare quick tunnels get reaped after idle periods — is
    treated as None so the email falls back to localhost links instead of
    rendering buttons that lead to "Site can't be reached".
    """
    if not tunnel_url_path.exists():
        return None
    text = tunnel_url_path.read_text().strip()
    if not text or not text.startswith("https://"):
        return None
    if not probe:
        return text
    if _tunnel_alive(text):
        return text
    log.warning("tunnel URL on disk (%s) is dead — falling back to localhost", text)
    return None


def write_tunnel_url(url: str, tunnel_url_path: Path) -> None:
    tunnel_url_path.parent.mkdir(parents=True, exist_ok=True)
    tunnel_url_path.write_text(url + "\n")


def _tunnel_alive(url: str) -> bool:
    """Probe the public URL to see if Cloudflare still routes it.

    A live quick tunnel returns 200/404/etc from the upstream Flask app. A
    reaped/dead tunnel returns either a Cloudflare error page (1033/1016) OR
    fails DNS resolution outright.
    """
    if not url:
        return False
    try:
        r = requests.get(url + "/", timeout=HEALTH_TIMEOUT_SEC, allow_redirects=False)
        # 5xx from Cloudflare with a `cf-mitigated` or `server: cloudflare` header
        # usually means the tunnel is dead even if HTTP is technically responsive.
        if 500 <= r.status_code < 600 and "cloudflare" in (r.headers.get("server", "").lower()):
            log.info("health: cf %d at %s — treating as dead", r.status_code, url)
            return False
        return True
    except Exception as e:
        log.info("health: probe failed for %s: %s", url, e)
        return False


def run_tunnel_forever(local_port: int, tunnel_url_path: Path) -> int:
    """Long-running entry point used by the launchd job.

    Spawns cloudflared, watches its output for the public URL, and rewrites
    `tunnel_url_path` whenever the URL changes. A background health-check
    thread periodically probes the URL; if it's been dead for
    HEALTH_FAILS_BEFORE_RESTART consecutive checks, we kill cloudflared so
    launchd respawns us with a fresh tunnel.
    """
    # --protocol http2 forces TCP (port 443) instead of QUIC (UDP/7844). Many
    # corporate / café networks block outbound UDP, leaving cloudflared in a
    # QUIC-retry loop that masquerades as "tunnel created" but never serves traffic.
    cmd = ["cloudflared", "tunnel", "--url", f"http://localhost:{local_port}",
           "--no-autoupdate", "--protocol", "http2"]
    log.info("starting cloudflared: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    state = {"last_url": None, "fails": 0, "stop": False}

    def _health_loop() -> None:
        # Give cloudflared ~30s to come up before first probe.
        time.sleep(30)
        while not state["stop"]:
            url = state["last_url"]
            if url:
                if _tunnel_alive(url):
                    if state["fails"]:
                        log.info("tunnel recovered")
                    state["fails"] = 0
                else:
                    state["fails"] += 1
                    log.warning("tunnel health failure %d/%d for %s",
                                state["fails"], HEALTH_FAILS_BEFORE_RESTART, url)
                    if state["fails"] >= HEALTH_FAILS_BEFORE_RESTART:
                        log.error("tunnel dead — killing cloudflared so launchd respawns it")
                        try: proc.terminate()
                        except Exception: pass
                        return
            time.sleep(HEALTH_INTERVAL_SEC)

    t = threading.Thread(target=_health_loop, daemon=True)
    t.start()

    try:
        if proc.stdout is None:
            return 1
        for line in proc.stdout:
            print(line, end="")  # forward to launchd stdout log
            m = _URL_RE.search(line)
            if not m:
                continue
            url = m.group(0)
            if url != state["last_url"]:
                write_tunnel_url(url, tunnel_url_path)
                log.info("tunnel URL updated: %s", url)
                state["last_url"] = url
                state["fails"] = 0
        proc.wait()
        return proc.returncode or 0
    except KeyboardInterrupt:
        proc.terminate(); proc.wait(timeout=5); return 0
    finally:
        state["stop"] = True
