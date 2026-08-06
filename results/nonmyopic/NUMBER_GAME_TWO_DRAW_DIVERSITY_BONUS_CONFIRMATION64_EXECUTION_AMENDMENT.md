# Diversity-Bonus Confirmation-64 Execution Amendment

Date frozen: 2026-08-06, before either prospective block seed was opened.

This amendment hardens execution only. It does not change the frozen selector,
supports, seeds, endpoints, bootstrap, scientific gates, or interpretation in
`NUMBER_GAME_TWO_DRAW_DIVERSITY_BONUS_PROSPECTIVE_PREREGISTRATION.md`.

## Exact Schedule

- shared run ID:
  `number-game-two-draw-diversity-bonus-confirmation64-20260808`;
- shared run directory:
  `results/nonmyopic/number_game_two_draw_diversity_bonus_confirmation64/number-game-two-draw-diversity-bonus-confirmation64-20260808`;
- Block A: 2026-08-08 Europe/London, ledger
  `results/nonmyopic/openrouter_daily_budget/2026-08-08.json`;
- Block B: 2026-08-09 Europe/London, ledger
  `results/nonmyopic/openrouter_daily_budget/2026-08-09.json`.

The formal executor is
`scripts/number_game_two_draw_diversity_bonus_confirmation64_daily_execute.py`.
Run it with `--block a` on August 8 and `--block b` on August 9. It creates the
day's ledger from live cumulative account credits and usage immediately before
the first formal call. Any account usage after that opening is charged against
the same `$5.00` cap. An existing nonempty formal block without a completed
daily record is never rerun. Immediately before model entry, the ledger moves
from `authorized_pending` to `execution_started`; that state is also never
reused after an ambiguous process crash.

## Independent Block-A Authorization

After Block A's measured spend is checkpointed, an independent zero-call
verifier recomputes only:

- the preregistration hash;
- source interface and exact tree, target, validation, and bootstrap seeds;
- exact `3,680` accepted requests;
- all source mechanics gates and their exact stage copy; and
- source result, tree, target, and private-response hashes.

It writes `BLOCK_A_AUTHORIZATION_VERIFICATION.json`. The authorization output
contains no Brier, comparison, selected root, target endpoint, or scientific
gate. Block B reruns this verifier before source construction and records the
exact verification-artifact hash in `BLOCK_B_STAGE.json`. The final combined
verifier binds that same pre-B hash.

Block B remains mandatory after a verified mechanics-clean Block A regardless
of Block A science. A failed Block A mechanics or authorization check forbids
Block B. Neither day authorizes a confirmation-related paid tail.

## Formal Commands

```bash
set -a
source .env
set +a

# 2026-08-08 Europe/London
python scripts/number_game_two_draw_diversity_bonus_confirmation64_daily_execute.py \
  --block a

# 2026-08-09 Europe/London
python scripts/number_game_two_draw_diversity_bonus_confirmation64_daily_execute.py \
  --block b
```

The August 7 history-blind control must already have a verified daily execution
record. The budget-model tail outcome is not a prerequisite and cannot affect
either diversity block.

## Zero-Call Integration Evidence

Before either formal directory existed, two temporary copies of the real
32-tree source artifact were relabeled only to the frozen A/B seed manifests.
The complete pipeline independently verified Block A, reconstructed all 64
fixed-selector rows, ran the 20,000-sample combined bootstrap, serialized and
reloaded the result, and passed all 37 final verification checks. This used
zero model calls and is mechanics evidence only.
