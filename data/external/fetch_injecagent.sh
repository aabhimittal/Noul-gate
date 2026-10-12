#!/usr/bin/env bash
# Fetch InjecAgent test cases (github.com/uiuc-kang-lab/InjecAgent) at a pinned commit.
# The repo has no LICENSE file (checked 2026-10-12), so the cases are fetched, not committed.
#   bash data/external/fetch_injecagent.sh && noulgate dataset import injecagent data/external/injecagent/*.json --out data/external/injecagent.jsonl
set -euo pipefail
COMMIT=f19c9f2c79a41046eb13c03c51a24c567a8ffa07
DIR="$(cd "$(dirname "$0")" && pwd)/injecagent"
mkdir -p "$DIR"
for f in test_cases_dh_base.json test_cases_ds_base.json; do
  curl -fsSL "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/$COMMIT/data/$f" -o "$DIR/$f"
done
echo "fetched into $DIR"
