# 20 Questions Target-Free Proxy Validation

**Status:** completed exploratory gate. The proposed target-free
support-dynamics policy signal is rejected.

The seed-1305 validation repeated the same 10-state, three-candidate screen
with non-thinking Gemma 4 26B A4B, now logging target-free support scores from
the current belief, likelihood rows, and production branch supports.

## Result

- The mechanism persists: mean within-state hidden truth-coverage spread is
  `0.1424`, maximum `0.5017`, and immediate EIG leaves at least `0.20`
  coverage on the table in `3/10` states.
- Immediate EIG has weak negative candidate-level rank association with hidden
  truth coverage: Spearman `-0.1061`.
- Expected current-support retention is not better: Spearman `-0.0810`.
- Expected surviving MAP mass is also not better: Spearman `-0.1283`.

The proxy therefore does not supply an actionable substitute for hidden truth
coverage and is not promoted to a `d2` policy score.

## Why It Fails

The target was absent from the current hypothesis support in `7/10` states.
For such a state, a support-retention score can only preserve the model's
current wrong hypotheses; it cannot know which question will cause a later
belief-generation call to rediscover the missing target. In the Wolverine
state, for example, all target-free retention choices had zero hidden coverage,
while `Is it a carnivore?` regenerated the target with expected coverage
`0.5017`.

This identifies the bottleneck precisely: the current natural-language 20
Questions system lacks reliable target recall in its belief support. A
non-myopic policy layered over that support would be an unvalidated proxy
optimization, not evidence for sequential BED. Do not run a headline `d2`
versus `d1` policy comparison in this environment without first repairing and
validating belief recall.

## Mechanics and Cost

- 10 states / 30 candidate rows; all target fields remain measurement-only.
- 5,344 requests, zero reasoning tokens, one length finish.
- Run cost: `$0.10137151` of the `$0.50` cap.
- Raw data: `results/nonmyopic/animals_coverage_dynamics/20260718_proxy_validation/COVERAGE_PROBE.json`.
