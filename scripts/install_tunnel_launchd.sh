#!/usr/bin/env bash
# Install the cloudflared quick-tunnel launchd job.
# Starts at boot, keeps cloudflared alive, writes the current public URL into
# data/tunnel_url.txt. Survives URL rotation by re-parsing cloudflared's output.
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$PROJECT/.venv/bin/python"
PLIST_SRC="$PROJECT/scripts/tunnel.plist.template"
PLIST_DST="$HOME/Library/LaunchAgents/local.news-digest-tunnel.plist"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "ERROR: cloudflared not on PATH. Install: brew install cloudflared" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: venv missing. Run scripts/bootstrap.sh first." >&2
  exit 1
fi

sed -e "s|__PYTHON__|$PYTHON|g" -e "s|__PROJECT__|$PROJECT|g" "$PLIST_SRC" > "$PLIST_DST"

launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load "$PLIST_DST"

echo "Installed tunnel: $PLIST_DST"
echo "Status:  launchctl list | grep news-digest-tunnel"
echo "URL:     cat $PROJECT/data/tunnel_url.txt   # (wait ~5s after install)"
echo "Logs:    tail -F $PROJECT/logs/tunnel.std*.log"
