#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

RUN_ID="number-game-full-retention-depth-three-powered-20260728"
OUTPUT_DIR="results/nonmyopic/number_game_full_retention_depth_three_powered/${RUN_ID}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite frozen run output: $OUTPUT_DIR" >&2
  exit 1
fi

exec "$PYTHON_BIN" scripts/number_game_full_retention_depth_three.py \
  --output-dir "$OUTPUT_DIR" \
  --run-id "$RUN_ID"
