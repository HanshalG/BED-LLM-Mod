#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

set -a
source "$REPO_ROOT/.env"
set +a

RUN_ID="longvid-four-hop-support-smoke-v2-$(date -u +%Y%m%dT%H%M%SZ)"

/opt/anaconda3/bin/python scripts/longvid_four_hop_support_smoke.py \
  --config configs/config_longvid_four_hop_support_smoke_openrouter.yaml \
  --output-dir "results/nonmyopic/longvid_four_hop_support_smoke/$RUN_ID" \
  --private-raw-dir results/nonmyopic/longvid_four_hop_support_smoke/private \
  --run-id "$RUN_ID"

