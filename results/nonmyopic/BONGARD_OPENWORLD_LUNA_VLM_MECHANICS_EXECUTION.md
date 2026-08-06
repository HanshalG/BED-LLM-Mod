# Bongard-OpenWorld Luna VLM Mechanics Execution

Pre-call interface note: the semantic-validity amendment removes history
double-conditioning and role markers, makes branch sensitivity unobserved-only,
and expands mechanics to at most 108 requests under a `$1.75` cap. The
selected-policy-only 88-request details below document the earlier zero-call
fixture and are not executable evidence.

The shuffled-control amendment corrects continuation-value permutation and
bumps mechanics to interface v3. Serving remains interface v2. The executable
command is the one-command wrapper documented
in `BONGARD_OPENWORLD_LUNA_AUG10_EXECUTION.md`; do not run the older component
commands independently.

Prepared: 2026-08-06
Earliest execution: 2026-08-10 Europe/London

## Exact-10 Command

```bash
set -a; source .env; set +a
python scripts/bongard_openworld_luna_vlm_serving_smoke.py \
  --output-dir results/nonmyopic/bongard_openworld_luna_vlm_serving_smoke/bongard-openworld-luna-vlm-serving-smoke-20260810 \
  --run-id bongard-openworld-luna-vlm-serving-smoke-20260810 \
  --daily-ledger results/nonmyopic/openrouter_daily_budget/2026-08-10.json
```

The wrapper creates the Aug 10 ledger from authenticated live account totals if
it does not exist. If it exists, the wrapper requires the same London date,
exact $5 cap, and enough account-wide remaining allowance. It refuses any
second execution recorded under the same interface.

The run is exactly ten nonreasoning Luna requests, projected at $0.10 and
capped at $0.25. Any exception is banked as `FAILURE.json` and live posted
spend is reconciled into the daily ledger before the exception returns.

## Frozen Inputs

- source/protocol manifest SHA:
  `7acd3cc9abd24fb60f7da98710aa2ed89b75d9c137ada46380f258d16380e763`
- image-integrity manifest SHA:
  `239943ae789ebdc2c0a03577a02b04890c6d00f50ce45639c5defc1624ccee96`
- archive SHA:
  `5ab838cffd8c1be6080e232b9fa4d9ca824d213371625abe59d984745f694d25`
- model: `openai/gpt-5.6-luna`
- model seed: `2026081001`
- reasoning: disabled
- image encoding: max 512 px, JPEG quality 88, high detail
- hypotheses per response: 10
- likelihoods per hypothesis: 14 integer probabilities

The ten assembled requests total 8,822,288 local JSON bytes. This is a transport
size measurement, not an API token estimate.

## Zero-Call Verification

```bash
pytest -q \
  tests/test_bongard_openworld_vlm_bed.py \
  tests/test_bongard_openworld_luna_vlm_serving_smoke.py \
  tests/test_bongard_openworld_source_protocol_audit.py \
  tests/test_bongard_openworld_image_integrity_audit.py
```

Current result: 25 passed. A full fixture replay using the real four mechanics
tasks also passes all 14 serving gates with zero model calls and zero cost.

Do not run the full four-task tree unless `RESULT.json` is `passed`, every gate
is true, and observed cost projects the full run below both its $1.75 cap and
the ledger's remaining account-wide allowance.

## Full Mechanics Command

Run on the same London day only after the exact-10 command passes:

```bash
set -a; source .env; set +a
python scripts/bongard_openworld_luna_vlm_mechanics_tree.py \
  --output-dir results/nonmyopic/bongard_openworld_luna_vlm_mechanics_tree/bongard-openworld-luna-vlm-mechanics-tree-20260810 \
  --run-id bongard-openworld-luna-vlm-mechanics-tree-20260810 \
  --serving-result results/nonmyopic/bongard_openworld_luna_vlm_serving_smoke/bongard-openworld-luna-vlm-serving-smoke-20260810/RESULT.json \
  --daily-ledger results/nonmyopic/openrouter_daily_budget/2026-08-10.json
```

The full wrapper independently replays the exact-10 raw responses, metrics,
and gates before spend. It then reserves the $1.75 cap against the account-wide
ledger and also refuses if `1.5 * observed serving cost/request * 108` exceeds
that cap. The first stage is exactly 68 shared requests. Final calls are
deduplicated across all realized first-action continuations and policy paths,
for at most 108 total requests.

A zero-call replay using all four real mechanics tasks produces 14 distinct
final histories and 82 total requests under the corrected control, passes
every frozen mechanics gate, and keeps development, confirmation, and
sealed-test access false. Its synthetic endpoint values are fixture behavior
and carry no scientific evidence.
