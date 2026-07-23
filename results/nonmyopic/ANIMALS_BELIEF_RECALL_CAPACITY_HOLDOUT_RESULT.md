# Animals Capacity-Gated Belief-Recall Holdout Result

Status: **scientific gate failed; capacity-gated line stopped**.

## Preregistered Result

All 60 fresh distinct-target states completed, and 19 states had nonzero
candidate coverage spread. The capacity-gated selector used the belief-recall
ranker only when the current support size was at most the configured generation
capacity, `max_num_samples=16`; otherwise it selected immediate EIG.

| Selector | Selected expected truth coverage | Mean regret | Active-state regret |
| --- | ---: | ---: | ---: |
| Immediate EIG | .081918 | .098204 | .310118 |
| Capacity-gated ranker | .091671 | .088451 | .279320 |
| Candidate oracle, measurement only | .180122 | 0 | 0 |

The paired gain was `+.009753` with `1/59/0` wins/ties/losses. The
preregistered producer interval was `[0, +.029258]`; its lower bound was not
strictly positive, so the all-pass gate failed.

## Independent Audit

The independent seed-24277 bootstrap reproduced the same mean and interval,
`[0, +.029258]`. It also verified that:

- all 60 model-visible payloads rebuild exactly from the target-free allowlist;
- no payload contains target or expected-truth-coverage fields;
- all 60 raw ranker responses reparse exactly;
- the capacity-gated summary and full producer gate dictionary replay exactly;
- all 60 measurement targets are distinct.

Every integrity check passed. The audit reports failure only because the
independent scientific gate also failed.

## Mechanism

Only 3 of 60 current supports were at or below 16. This is distinct from the
19 states with nonzero candidate coverage spread reported as active by the
ranking summary. The capacity policy therefore used immediate EIG on 57 states
and changed EIG's selected candidate on only two:

- support size 16: one `+.585167` expected-coverage win;
- support size 14: one zero-effect tie.

The gate successfully removed the losses seen in the earlier ungated holdout,
but it also made the selector nearly identical to immediate EIG. The single
effective win is not enough to establish a reusable non-myopic policy.

For diagnosis only, the ungated ranker on these same records had mean gain
`+.030753` and `5/53/2` wins/ties/losses. This endpoint was inspected after the
capacity policy was frozen and is not a replacement test or evidence for
tuning a wider threshold.

## Serving And Cost

- Coverage producer: 32,184 requests, 4,712,722 prompt + 455,533 completion
  tokens, zero reasoning, `$0.55820292`.
- Ranker: 60 requests, 29,351 prompt + 1,370 completion tokens, zero reasoning,
  `$0.00323469`.
- Total holdout cost: `$0.56143761`.
- Project spend after holdout: `$41.64327541` of `$110`.

Raw producer and audit artifacts are in
`results/nonmyopic/animals_belief_recall_capacity_holdout/gemma26b_20260723/`.
