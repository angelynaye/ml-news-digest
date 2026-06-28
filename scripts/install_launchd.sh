#!/usr/bin/env bash
# Install the news-digest launchd job. Reads schedule from config.yaml.
# Idempotent: unloads any existing job first.
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$PROJECT/config.yaml"
PLIST_SRC="$PROJECT/scripts/launchd.plist.template"
PLIST_DST="$HOME/Library/LaunchAgents/local.news-digest.plist"
PYTHON="$PROJECT/.venv/bin/python"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: $CONFIG missing. Copy config.example.yaml to config.yaml first." >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: venv missing at $PYTHON. Run scripts/bootstrap.sh first." >&2
  exit 1
fi

HOUR=$("$PYTHON" -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['schedule']['send_hour'])")
MINUTE=$("$PYTHON" -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['schedule']['send_minute'])")
AWS_PROFILE="${AWS_PROFILE:-claude}"
AWS_REGION="${AWS_REGION:-us-east-1}"

echo "Installing launchd job at ${HOUR}:$(printf %02d $MINUTE) local time…"

sed \
  -e "s|__PYTHON__|$PYTHON|g" \
  -e "s|__PROJECT__|$PROJECT|g" \
  -e "s|__HOUR__|$HOUR|g" \
  -e "s|__MINUTE__|$MINUTE|g" \
  -e "s|__AWS_PROFILE__|$AWS_PROFILE|g" \
  -e "s|__AWS_REGION__|$AWS_REGION|g" \
  "$PLIST_SRC" > "$PLIST_DST"

launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load "$PLIST_DST"

echo "Installed: $PLIST_DST"
echo "Next run: ${HOUR}:$(printf %02d $MINUTE) local time."
echo "To run now: launchctl start local.news-digest"
echo "To uninstall: scripts/uninstall_launchd.sh"
