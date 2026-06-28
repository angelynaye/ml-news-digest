#!/usr/bin/env bash
# One-shot setup: create venv, install deps, copy config.
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -r requirements.txt

if [[ ! -f "config.yaml" ]]; then
  cp config.example.yaml config.yaml
  echo "Created config.yaml. Edit it to set recipients, SMTP, and schedule."
fi

mkdir -p data logs
echo "Bootstrap complete."
echo
echo "Next steps:"
echo "  1) Edit config.yaml — recipients, SMTP host/user/from, schedule.send_hour."
echo "  2) Store SMTP password in Keychain (NOT in the config):"
echo "       security add-generic-password -a \"\$USER\" -s news-digest-smtp -w 'YOUR_APP_PASSWORD'"
echo "  3) Test fetch only:     ./.venv/bin/python -m news_digest.main fetch-only"
echo "  4) Test full preview:   ./.venv/bin/python -m news_digest.main preview"
echo "  5) Verify SMTP:         ./.venv/bin/python -m news_digest.main test-smtp"
echo "  6) Install daily cron:  ./scripts/install_launchd.sh"
