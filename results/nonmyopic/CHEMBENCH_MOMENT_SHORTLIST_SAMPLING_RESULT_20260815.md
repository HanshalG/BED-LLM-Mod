# ChemBench Moment-Shortlist Sampling Result

Date: 2026-08-15 (Europe/London)

## Binding

- Result: `results/nonmyopic/chembench_moment_shortlist_sampling/result.json`
- Result SHA-256:
  `d93c1929d17ec285761e2fb881d31f11f3d49db59cdb88c4145de2b927bfad57`
- Protocol SHA-256:
  `ac30b9c79527976cf86939232b317d9d3769dc3fd7608370cd760674d6167266`
- Implementation SHA-256:
  `4c317c4d6083caf41609be8a3686e74001b3d92fa495b49f65bcbc48456247c1`
- RQMC and IID predecessor SHA-256:
  `a966d5cf4984c9907649a0dae5d6bb8a19982f942c83f463f7ec61e4e2d439f2`
  and `d71964ff247bc80408b0b5c78c4b168c5117ece372446990c4a5dfd17b42c0f6`.

The complete 36-case run finished in 24.84 seconds with no LLM, API, network
call, endpoint, or paid resource.

## Frozen Decision

The endpoint-blind shortlist passed all pruning gates, but shared-IID sampling
failed rank coverage. The complete current-root ChemBench route is closed.

### Shortlist

- Full-reference best included: 91.67% (required at least 90%).
- Cases with shortlist-oracle regret at most 3%: 100%.
- Mean shortlist-oracle normalized regret: 0.0356% (required at most 0.5%).
- Median proxy/reference Spearman across all actions: approximately 0.729.

### Sampling

| 256 replicate | Median rho | Fraction rho >= .8 | Full regret <= 3% | Mean regret |
|---:|---:|---:|---:|---:|
| 1 | 0.8582 | 55.56% | 100% | 0.204% |
| 2 | 0.8582 | 58.33% | 100% | 0.159% |
| 3 | 0.8135 | 52.78% | 100% | 0.207% |
| 4 | 0.7951 | 50.00% | 100% | 0.148% |
| Required, each | >= 0.90 | >= 90% | >= 90% | <= 1% |

All four 256 replicates pass pooled and component-bank regret conditions. The
1,024-sample ensemble still has median rho 0.8503 and 58.33% rank coverage.

## Interpretation

The moment proxy is a useful action generator but does not solve sampled
full-list ranking. Across four successive representations, good-action regret
has been consistently much easier than high Spearman coverage. The current
correct-structure panel contains many nearly equivalent action values and does
not provide a clean first link for a sampled non-myopic planner under the
frozen workshop-strength gate.

Per protocol, do not tune shortlist size, proxy terms, seeds, sample count, or
rank thresholds on these cases. Retain the moment proxy as an optional action
generator, but move primary development to a prospectively designed
compositional structure-discovery environment with a stronger diagnostic-
then-targeted horizon gap. The next gate must establish that gap exactly or
with high-sample oracle computation before any LLM call.

No depth or LLM claim is authorized by this result.
