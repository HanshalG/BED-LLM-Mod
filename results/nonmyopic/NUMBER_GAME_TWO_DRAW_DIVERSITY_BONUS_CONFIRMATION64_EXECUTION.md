# Number Game Diversity-Bonus Confirmation-64 Execution

Date frozen: 2026-08-06

The later execution amendment makes the exact formal path authoritative:

```bash
# 2026-08-08 Europe/London
python scripts/number_game_two_draw_diversity_bonus_confirmation64_daily_execute.py --block a

# 2026-08-09 Europe/London
python scripts/number_game_two_draw_diversity_bonus_confirmation64_daily_execute.py --block b
```

It fixes the shared run ID and paths, initializes each account-wide ledger
from live cumulative usage, independently verifies Block A before Block B, and
refuses partial block reruns. The placeholder commands below describe the
underlying component interface and must not be launched separately.

## Preconditions

Run only after the sealed August 7 history-blind control is complete and
banked. Each block requires its own newly initialized Europe/London daily
ledger with a `$5.00` cap, zero prior spend, at least `$5.00` live balance,
and no second confirmation block authorized. A separately frozen unrelated
tail may start only after the confirmation block's measured and posted spend
are reconciled, and its hard cap must fit the exact remaining allowance.

Block B must use a strictly later London date than Block A. It is mandatory
whenever Block A mechanics pass and is forbidden when they fail. No Block A
scientific value enters authorization.

## Block A

```bash
set -a
source .env
set +a

/Users/hanshalgoyal/.conda/envs/20_questions_env/bin/python \
  scripts/number_game_two_draw_diversity_bonus_confirmation64_staged.py \
  --run-dir <fresh-shared-run-directory> \
  --run-id <fresh-run-id> \
  --block a \
  --daily-ledger <block-a-current-day-ledger.json>
```

Block A uses trees `110000..110031`, targets `110100..110131`, validation
start `110200`, and source bootstrap `110800`. Its public stage record is a
mechanics-only authorization with no Brier, comparison, selected-root, or
bonus-policy result.

## Block B

On a later London date, regardless of any Block A scientific values:

```bash
set -a
source .env
set +a

/Users/hanshalgoyal/.conda/envs/20_questions_env/bin/python \
  scripts/number_game_two_draw_diversity_bonus_confirmation64_staged.py \
  --run-dir <same-shared-run-directory> \
  --run-id <same-run-id> \
  --block b \
  --daily-ledger <block-b-current-day-ledger.json>
```

Block B uses trees `111000..111031`, targets `111100..111131`, validation
start `111200`, and source bootstrap `111800`. It writes the sole combined
`RESULT.json`, using all 64 paired rows and 20,000 bootstrap resamples with
seed `112800`.

## Workload And Gates

- model: `qwen/qwen3.7-plus`, nonreasoning;
- two independent support draws per planning history;
- exactly `3,680` accepted requests and maximum `$5.00` per block;
- exactly `7,360` requests and maximum `$10.00` total;
- fresh score calculations: coefficients `0.0` and frozen `-0.5` only;
- no coefficient grid published or retained.

Primary gates are at least 3% Brier reduction versus dynamic-support depth
two, a paired 64-tree bootstrap interval below zero, and wins exceeding
losses. Co-required checks are at least 16 roots changed from unadjusted depth
three and mean bonus Brier not worse than unadjusted depth three.

## Verification

After measured Block B spend is checkpointed, the runner executes
`scripts/number_game_two_draw_diversity_bonus_confirmation64_verify.py`.
It independently binds actual and protocol seeds, both stage source hashes
including target artifacts, later-day mechanics-only authorization, exact
request counts, all 64 fixed-selector rows, all comparisons and rank metrics,
the combined bootstrap, every gate, and final status.

`BLOCK_B_DAILY_EXECUTION.json` is written only after verification succeeds
and posted usage is reconciled. A verifier failure leaves spend recorded and
fails closed.

## Dry Replay Evidence

A final zero-call end-to-end mechanics replay used two temporary copies of the
hash-bound historical 32-tree source, relabeled only for the two frozen seed
protocols and dates. It reconstructed 64 rows, produced the combined result,
survived JSON serialization/reload, and passed all `35` verifier checks. This
is a mechanics test, not scientific evidence.
