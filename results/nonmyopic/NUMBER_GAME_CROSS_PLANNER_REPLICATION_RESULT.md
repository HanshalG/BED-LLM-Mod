# Number Game Cross-Planner Replication Result

Date: 2026-07-28

Protocol:
`NUMBER_GAME_CROSS_PLANNER_REPLICATION_PREREGISTRATION.md`

Run:
`results/nonmyopic/number_game_cross_planner_replication/number-game-cross-planner-replication-20260728T160000Z`

## Verdict

**Formal preregistered conjunction fails, with strong directional model-family
robustness.**

GPT-5.4 Mini replaced Gemini as the generator of the initial and
branch-conditioned planning supports; Gemini generated independent target
concepts. Predictive-risk BED has wholly negative whole-tree Brier intervals
against every baseline and comfortably passes the PTS, fixed-support, random,
coverage, novelty, transport, and reasoning gates.

It narrowly misses three frozen myopic magnitude/count criteria:

- Brier gain is `9.9928%` rather than at least `10%`;
- Brier wins are `23/32` rather than at least `24/32`; and
- Hamming reduction is `12.80%` rather than at least `15%`.

Three branch cells retain seven valid unique answer-consistent rules rather
than the required eight. No threshold was changed, tree dropped, or branch
regenerated after seeing the result.

## Results

| Baseline | Candidate Brier | Baseline Brier | Relative gain | Brier wins | Whole-tree 95% difference CI |
|---|---:|---:|---:|---:|---:|
| myopic EIG | 0.18898 | 0.20996 | 9.99% | 23/32 | [-0.03227, -0.01106] |
| fixed-support depth two | 0.18898 | 0.20998 | 10.00% | 23/32 | [-0.03177, -0.01147] |
| exact uniform random root | 0.18898 | 0.22261 | 15.11% | 30/32 | [-0.04150, -0.02596] |
| seeded PTS | 0.18898 | 0.22672 | 16.65% | 29/32 | [-0.05320, -0.02376] |

Predictive-risk BED selects a different first query from myopic and
fixed-support depth two on 30/32 trees.

Best-rule Hamming error is `0.11578`:

- versus myopic `0.13278`, a `12.80%` reduction with difference CI
  `[-0.03205, -0.00347]`;
- versus fixed depth two `0.13484`, a `14.13%` reduction; and
- versus PTS `0.14783`, a `21.68%` reduction with CI
  `[-0.04727, -0.01847]`.

Mean exact-extension coverage rises by `1.59` percentage points versus
myopic, `1.83` points versus fixed depth two, `5.98` points versus random, and
`6.21` points versus PTS.

For target extensions absent from planning support, candidate-minus-myopic
mean differences are `-0.01436` Brier and `-0.01960` Hamming. The candidate
wins 18/32 novel-target tree means on Brier and 21/32 on Hamming.

## Structural Failure

All 32 trees have at least 20 valid initial rules, at least 21 target rules,
and at least eight target extensions novel to planning support. Three of 512
branch cells have seven valid rules:

- tree seed `27000`, root `10`, negative answer;
- tree seed `27019`, root `46`, positive answer; and
- tree seed `27021`, root `14`, negative answer.

These responses contained 24 syntactically valid rules, but 14--17 contradicted
the stipulated branch answer and were correctly filtered. The aggregate
includes all trees unchanged.

## Accounting

- Successful responses / HTTP attempts: `576 / 576`
- Retries / provider-error retries: `0 / 0`
- Reasoning tokens / forced exits: `0 / 0`
- Cost: `$1.6033413`
- Result SHA-256:
  `1eeeb607455fba9b594587f27c3283974920a591e3ab6ddc35f435bb066ac67d`
- Tree artifact SHA-256:
  `a4b0d8b52c2420827d76a8bd38bbdf630c4151c0da868ab07b61b3f7696ea1fb`
- Private raw-response SHA-256:
  `3b68855dd1254a7b1ae498e0503e98e0ac17aeb046fe8cdd9799ebdf9fbffa31`
