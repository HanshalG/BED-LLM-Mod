# Animals Branch-Support Ranker Development Result

Status: **development gate failed; no fresh holdout**.

## Result

All 20 seed-24279 states and 60 candidate branches completed. Every stored
model payload rebuilt exactly from the target-free allowlist and excluded the
target field, truth coverage, branch truth indicators, and immediate EIG.

| Selector | Selected expected truth coverage |
| --- | ---: |
| Immediate EIG | .025000 |
| Branch-support ranker | .050500 |
| Candidate oracle, measurement only | .080792 |

The paired gain was `+.025500` with `1/19/0` wins/ties/losses. However,
candidate-level Spearman association remained negative: `-.086489` for the
branch-support ranker versus `-.100543` for immediate EIG. The frozen gate
required positive rank association as well as a better top choice, so the
exact scorer fails and no fresh holdout is authorized.

## Mechanism

The ranker changed EIG's selected candidate in 10 states. Nine changes were
zero-coverage ties. The one consequential change occurred when Meerkat was
absent from the current support: the ranker selected `Is it larger than a
medium-sized dog?`, whose generated branches gave expected truth coverage
`.51`, while EIG's Australia question gave zero.

Actual branch contents can therefore reveal an open-world recovery opportunity
that is invisible to current-support scoring. But only 3 of 20 targets appeared
anywhere in the six counterfactual supports, and only three states had nonzero
candidate coverage spread. The ranker signal is too sparse and globally
miscalibrated to justify confirmatory spending.

## Serving And Cost

- Coverage: 10,012 requests, 1,476,146 prompt + 142,563 completion tokens,
  zero reasoning, `$0.17735102`.
- Ranker: 20 requests, 17,892 prompt + 479 completion tokens, zero reasoning,
  `$0.00179433`.
- Total: `$0.17914535`.
- Project spend: `$42.00420333` of `$110`.

Artifacts are in
`results/nonmyopic/animals_branch_support_ranker_development/gemma26b_seed24279/`.
