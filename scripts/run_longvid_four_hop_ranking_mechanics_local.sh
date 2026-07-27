#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

set -a
source "$REPO_ROOT/.env"
set +a

RUN_ID="longvid-four-hop-ranking-mechanics-$(date -u +%Y%m%dT%H%M%SZ)"

/opt/anaconda3/bin/python scripts/longvid_four_hop_ranking_mechanics.py \
  --config configs/config_longvid_four_hop_ranking_mechanics_openrouter.yaml \
  --qa-path "/tmp/longvidsearch.STXBt8/full-QA(3000).json" \
  --caption-path /tmp/longvidsearch.STXBt8/video-caption.parquet \
  --output-dir "results/nonmyopic/longvid_four_hop_ranking_mechanics/$RUN_ID" \
  --private-raw-dir results/nonmyopic/longvid_four_hop_ranking_mechanics/private \
  --run-id "$RUN_ID"

