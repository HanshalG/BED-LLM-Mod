# COPEx Direct-Proposal Quadrature Pilot Result

The quadrature pilot completed at
`results/nonmyopic/copex_direct_proposals_quadrature_pilot/20260718/FACTORIAL.json`.

## Validity

- 8 paired trials, 8 queries, fresh seed 46022.
- Full finite-support three-node Gaussian quadrature scored every immediate and child
  candidate. Eight stratified source/noise child branches were used only to obtain
  future-state LLM proposal cells.
- All moves legal; initial d1/d2 root cell shared; width's state-local logical call
  allocation matched d2; zero rejected cells and zero reasoning tokens.
- 2,777 accepted physical cells; `$0.12298039` total cost.

## Result

Positive entropy-AUC gains favor depth two.

| Comparison | Mean gain (nats) | 95% paired bootstrap CI | W / T / L |
| --- | ---: | --- | --- |
| LLM d2 - shared LLM d1 | -0.0374 | [-0.3496, +0.2373] | 5 / 1 / 2 |
| LLM d2 - LLM matched width | -0.0870 | [-0.3779, +0.2280] | 3 / 1 / 4 |
| Grid d2 - grid d1 | +0.1474 | [+0.0095, +0.3305] | 6 / 1 / 1 |
| LLM d2 - grid d2 | +0.1809 | [-0.2646, +0.6138] | 6 / 0 / 2 |

The LLM d2 versus d1 true-particle log-posterior AUC contrast is `-0.0693` nats
(`[-0.4459, +0.2136]`). The depth-by-proposal interaction is `-0.1847` nats
(`[-0.6265, +0.1342]`).

## Diagnosis and Decision

This pilot fails promotion: d2 does not beat either LLM d1 or LLM width, and the
truth-log-posterior cross-check is negative. The numerical estimator is not the
explanation, because grid d2 has a positive paired depth advantage under the same
quadrature score. The d2 opportunity exists, but non-thinking LLM child proposal
cells are too generic: 94.6% of emitted angles are cardinal/diagonal defaults, and
the same three-angle sequences recur across distinct child beliefs. The next limited
screen tests whether a bounded 26B thinking proposer corrects that proposal-quality
failure before it is admitted to another d2 policy run.
