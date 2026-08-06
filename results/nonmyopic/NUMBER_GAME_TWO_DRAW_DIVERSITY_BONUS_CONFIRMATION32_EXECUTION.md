# Number Game Diversity-Bonus Confirmation-32 Execution

Date frozen: 2026-08-06

## Preconditions

Run only after the sealed August 7 history-blind control is complete and
banked. Before execution:

1. initialize a new Europe/London daily ledger for the execution date using
   the live OpenRouter cumulative usage as its opening baseline;
2. confirm the ledger has a `$5.00` cap, zero recorded spend, and no other
   paid block authorized;
3. confirm at least `$5.00` live balance; and
4. run the focused tests below from committed code.

The runner refuses before source construction if the preregistration hash,
ledger date, remaining allowance, predecessor hashes, or balance gate fails.

## Command

```bash
set -a
source .env
set +a

/Users/hanshalgoyal/.conda/envs/20_questions_env/bin/python \
  scripts/number_game_two_draw_diversity_bonus_confirmation32.py \
  --output-dir <fresh-output-directory> \
  --run-id <fresh-run-id> \
  --daily-ledger <current-day-ledger.json>
```

## Frozen Workload

- tree seeds: `110000..110031`;
- target seeds: `110100..110131`;
- validation seed start: `110200`;
- bootstrap seed/samples: `110800` / `20000`;
- exact accepted requests: `3680`;
- score: `z(depth-three predicted Brier) - 0.5 * z(mean future-branch
  two-draw Jaccard distance)`;
- coefficients calculated on fresh data: exactly `0.0` and `-0.5`; only the
  fixed adjusted scores are published;
- maximum daily cost: `$5.00`;
- additional paid blocks that day: none.

## Verification

The daily executor checkpoints locally measured spend before verification.
It then runs
`scripts/number_game_two_draw_diversity_bonus_confirmation32_verify.py`,
which independently:

- binds the actual source tree, target, validation, and bootstrap seeds;
- reconstructs all valid extension sets from the two raw Qwen draws;
- verifies that no coefficient grid was calculated or published;
- reproduces all 32 selected roots and adjusted scores;
- reproduces all paired comparisons and 20,000-sample bootstraps;
- recomputes the six scientific gates and final status; and
- writes `VERIFICATION.json` with zero model calls.

`DAILY_EXECUTION.json` is written only after verification succeeds and posted
account usage is reconciled. A verifier failure leaves the measured spend in
the ledger and fails closed.

## Tested Dry Replay

The complete path was exercised without provider calls by relabeling a copied
hash-bound 32-tree historical source in a temporary directory to the frozen
prospective seed protocol. This was a mechanics test only, not scientific
evidence. It reparsed every raw response, reran scoring and bootstrap,
serialized/reloaded the result, and passed all `25/25` independent checks.

The first dry replay correctly failed because JSON converted integer adjusted-
score keys to strings. Keys are now canonicalized before writing; the repeated
end-to-end replay verifies exactly. This demonstrates that the verifier is
capable of rejecting a plausible serialization mismatch rather than merely
rubber-stamping runner output.

## Focused Tests

```bash
pytest -q \
  tests/test_number_game_two_draw_diversity_bonus_audit.py \
  tests/test_number_game_two_draw_diversity_bonus_confirmation32.py \
  tests/test_number_game_two_draw_diversity_bonus_confirmation32_verify.py \
  tests/test_number_game_qwen_fully_fresh_daily_stages.py \
  tests/test_number_game_qwen_fully_fresh_control_daily_execute.py \
  tests/test_number_game_qwen_fully_fresh_control_verify.py \
  tests/test_number_game_budget_model_reliability128.py \
  tests/test_openrouter_daily_budget.py
```

Expected current result: `55 passed` or greater as focused tests are added.
