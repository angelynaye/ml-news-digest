#!/usr/bin/env bash
# Convenience wrapper used by both humans and launchd.
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT"
export PYTHONPATH="$PROJECT/src"
exec "$PROJECT/.venv/bin/python" -m news_digest.main "$@"
