# Animals Stratified-Prior Development Result

Status: **development gate failed; untouched holdout not run**.

The shared eight-component target prior and belief generator completed all 20
fixed development states. Truth appeared in 14/20 six-branch unions, passing
the union-coverage gate. However, 13/20 truths were already present in the
initial broad support, and only one initially omitted truth was recovered by a
counterfactual branch, far below the required six.

| Selector | Selected expected truth coverage |
| --- | ---: |
| Immediate EIG | .419050 |
| Branch-content ranker | .366667 |
| Candidate oracle, measurement only | .620833 |

The paired gain was `-.052383` with `3/14/3` wins/ties/losses. Ranker Spearman
was `-.216711` versus EIG `-.273911`, and active-state regret was worse
(`.363095` versus `.288262`).

This closes the exact-name Animals formulation from both directions:

- narrow or tail-mismatched supports rarely recover the hidden truth;
- broad prior-matched supports contain the truth initially and remove the
  path-dependent recovery opportunity.

The untouched 60-target split remains unused and no confirmation is
authorized.

Usage: 28,330 coverage + 20 ranker requests, zero reasoning, `$0.52711392`.
Project spend is `$44.69379594` of `$110`.

Artifacts are in
`results/nonmyopic/animals_stratified_prior_development/gemma26b_seed24286/`.
