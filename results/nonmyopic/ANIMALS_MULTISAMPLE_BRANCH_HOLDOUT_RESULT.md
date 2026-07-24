# Animals Multi-Sample Branch-Ranker Holdout Result

Status: **scientific gate failed; exact fresh-target line stopped**.

All 60 fresh distinct-target states completed, but the four-list generator
recovered the target in only `6/60` six-branch unions. Only five states had
nonzero candidate coverage spread, far below both preregistered thresholds of
20.

| Selector | Selected expected truth coverage |
| --- | ---: |
| Immediate EIG | .022667 |
| Multi-sample branch ranker | .008472 |
| Candidate oracle, measurement only | .043139 |

The paired ranker-minus-EIG gain was `-.014194` with `0/58/2`
wins/ties/losses. The producer and independent seed-24283 intervals were both
`[-.038542, 0]`. Ranker candidate Spearman was `-.053538`, slightly above but
still negative relative to EIG's `-.083598`. Active-state regret was worse:
`.416000` versus `.245667`.

The independent audit verified all target-free payloads, raw response parses,
ranking summaries, union counts, four-call registration, producer gates, and
60 distinct targets. Integrity passes; the scientific result does not.

The development set contained more familiar animals and achieved `8/20` union
coverage. The new set was intentionally disjoint and substantially more
taxonomically obscure. The collapse to `6/60` shows that multi-sampling does
not solve exact-name recovery under target distributions far into the LLM
prior tail. This line cannot be repaired or rerun on these targets.

Serving:

- Coverage: 41,405 requests, 6,033,357 prompt + 614,846 completion tokens,
  zero reasoning, `$0.76115075`.
- Ranker: 60 requests, 61,810 prompt + 1,446 completion tokens, zero reasoning,
  `$0.00664354`.
- Total: `$0.76779429`.
- Project spend: `$44.16486280` of `$110`.

Artifacts are in
`results/nonmyopic/animals_multisample_branch_holdout/gemma26b_seed24281/`.
