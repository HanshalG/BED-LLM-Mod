#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

set -a
source .env
set +a

RUN_ID="number-game-crossfit-endpoint-precision-20260728"
OUTPUT_DIR="results/nonmyopic/number_game_crossfit_endpoint_precision/${RUN_ID}"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing output: $OUTPUT_DIR" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" scripts/number_game_crossfit_endpoint_precision.py \
  --run-id "$RUN_ID" \
  --output-dir "$OUTPUT_DIR"
