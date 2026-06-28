#!/usr/bin/env bash
# Install the news-digest server launchd job (Flask server for /click /up /down /audio).
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$PROJECT/.venv/bin/python"
PLIST_SRC="$PROJECT/scripts/server.plist.template"
PLIST_DST="$HOME/Library/LaunchAgents/local.news-digest-server.plist"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: venv missing. Run scripts/bootstrap.sh first." >&2
  exit 1
fi

sed -e "s|__PYTHON__|$PYTHON|g" -e "s|__PROJECT__|$PROJECT|g" "$PLIST_SRC" > "$PLIST_DST"

launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load "$PLIST_DST"

echo "Installed server: $PLIST_DST"
echo "Status:  launchctl list | grep news-digest-server"
echo "Logs:    tail -F $PROJECT/logs/server.std{out,err}.log"
echo "URL:     http://127.0.0.1:9123/"
