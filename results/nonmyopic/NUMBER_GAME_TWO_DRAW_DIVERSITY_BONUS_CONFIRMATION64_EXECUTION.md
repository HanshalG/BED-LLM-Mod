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
refuses partial block reruns. Verified Block B completion also writes the
fixed-format public result note and machine-readable claim report
automatically. Completed replay independently regenerates both reports and
refuses missing, altered, or hash-mismatched artifacts. The placeholder commands below
describe the underlying component interface and must not be launched
separately.

On a pristine block, the paid command automatically runs the complete
read-only preflight before creating the daily ledger or invoking the model
adapter. The command proceeds only from `ready_without_paid_calls` and freezes
the same authenticated credit snapshot that passed the gate as the ledger's
opening balance. Wrong-day, waiting-predecessor, catalog, protocol, path, and
balance failures therefore leave no execution artifact. Completed and
failed-closed blocks remain banked and are never submitted again.

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

The separately frozen claim plan keeps that registered status distinct from
causal mechanism evidence. Selector superiority additionally requires the
bonus-versus-unadjusted paired mean and 95% interval below zero plus wins
exceeding losses. The strongest LLM-native tier also requires at least 3%
dynamic-versus-fixed depth-three improvement, an interval below zero, and wins
exceeding losses. These stricter interpretation families cannot rescue or
alter the registered result status.

## Verification

After measured Block B spend is checkpointed, the runner executes
`scripts/number_game_two_draw_diversity_bonus_confirmation64_verify.py`.
It independently binds actual and protocol seeds, both stage source hashes
including target artifacts, later-day mechanics-only authorization, exact
request counts, all 64 fixed-selector rows, all comparisons and rank metrics,
the combined bootstrap, every gate, and final status.

Before either paid block, the read-only protocol preflight also binds
`NUMBER_GAME_TWO_DRAW_DIVERSITY_BONUS_CONFIRMATION64_CLAIM_PLAN.md` SHA-256
`27f494f4fa85412a4f0b35c37d40215ea4287cd70aee273a90017d85cbf982e3`.
After Block B, the report path independently replays the verifier, requires
exact equality with `VERIFICATION.json`, and banks `CLAIM_REPORT.json` plus
the public Markdown exactly once with zero model calls.

`BLOCK_B_DAILY_EXECUTION.json` is written only after verification succeeds
and posted usage is reconciled. A verifier failure leaves spend recorded and
fails closed.

## Dry Replay Evidence

A final zero-call end-to-end mechanics replay used two temporary copies of the
hash-bound historical 32-tree source, relabeled only for the two frozen seed
protocols and dates. It reconstructed 64 rows, produced the combined result,
survived JSON serialization/reload, and passed all `35` verifier checks. This
is a mechanics test, not scientific evidence.
