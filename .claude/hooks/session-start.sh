#!/bin/bash
# SessionStart hook for Claude Code cloud sessions: install everything the tests, ruff and the
# HelixDB store (embedded runtime) need. The [d1] extra (torch + a 6 GB model) is deliberately left out;
# install it by hand when you need the real model.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"
python3 -m pip install --quiet --disable-pip-version-check -e ".[serve,mcp,report,dev,helix]"
echo "noulgate: dependencies installed (HelixDB store: embedded, see CLAUDE.md)" >&2
