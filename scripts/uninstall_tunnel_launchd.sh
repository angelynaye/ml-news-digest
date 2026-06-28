#!/usr/bin/env bash
set -euo pipefail
PLIST="$HOME/Library/LaunchAgents/local.news-digest-tunnel.plist"
launchctl unload "$PLIST" 2>/dev/null || true
rm -f "$PLIST"
echo "Uninstalled $PLIST"
