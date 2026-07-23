# Animals Belief-Recall Ranker Holdout Result

Status: **scientific gate failed; exact ungated ranker line stopped**.

## Preregistered Result

All 60 fresh distinct-target states completed, 24 had nonzero candidate
coverage spread, and the target-blind ranker was directionally better than
immediate EIG:

| Selector | Selected expected truth coverage | Mean regret | Active-state regret |
| --- | ---: | ---: | ---: |
| Immediate EIG | .120944 | .100632 | .251580 |
| Belief-recall ranker | .162207 | .059369 | .148424 |
| Candidate oracle, measurement only | .221576 | 0 | 0 |

The paired gain was `+.041263` with `8/47/5` wins/ties/losses. However, the
preregistered producer interval was `[-.017140, +.102098]`, so its lower bound
was not positive and the all-pass gate failed.

Candidate-level rank association was weak for both selectors: Spearman `.0087`
for the ranker and `-.0273` for immediate EIG. The top-choice effect was more
promising than global calibration, but not precise enough for a claim.

## Independent Audit

The independent seed-24273 bootstrap reproduced the same mean with interval
`[-.018110, +.099833]`. It also verified:

- all 60 model-visible payloads rebuild exactly from the target-free allowlist;
- no payload contains target or truth-coverage fields;
- all 60 raw ranker responses reparse to their stored scores;
- all summaries and producer gate decisions replay exactly;
- all 60 measurement targets are distinct.

The audit correctly reports failure because the independent lower bound also
crosses zero. This is a valid scientific null, not an artifact or serving
failure.

## Post-Hoc Mechanism

The five ranker losses cluster in larger current supports. The three largest
losses occurred at support sizes 20 or 23, where the semantic recall score
overrode a useful immediate-EIG candidate despite the support already having
expanded beyond one generation batch.

An explicitly post-hoc confidence gate is promising: apply the ranker only
when current support size is at most `max_num_samples=16`, otherwise use
immediate EIG. On this holdout it has `4/56/0` wins/ties/losses, mean gain
`+.041054`, and post-hoc interval `[+.008472,+.083610]`. This is not evidence
because the gate was selected after inspecting the holdout. It motivates one
new policy whose threshold is structurally defined by generation capacity and
must be tested on a wholly fresh target set.

## Serving And Cost

- Coverage producer: 31,859 requests, 4,661,847 prompt + 451,246 completion
  tokens, zero reasoning, `$0.56098143`.
- Ranker: 60 requests, 29,475 prompt + 1,409 completion tokens, zero reasoning,
  `$0.00339784`.
- Total holdout cost: `$0.56437927`.
- Project spend after holdout: `$41.08183780` of `$110`.

Raw producer and audit artifacts are in
`results/nonmyopic/animals_belief_recall_holdout/gemma26b_20260723/`.
