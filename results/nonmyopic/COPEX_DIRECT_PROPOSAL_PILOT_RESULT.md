# COPEx Direct-Proposal Monte Carlo Pilot Result

The boundary-amended direct-proposal pilot completed successfully at
`results/nonmyopic/copex_direct_proposals_pilot/20260718b/FACTORIAL.json`.

## Validity

- 8 paired trials, 8 sequential queries, seed 46021.
- All selected actions legal; initial LLM d1/d2 root cells shared; state-local width
  allocation matched d2's virtual child-cell allocation.
- 1,443 accepted physical LLM cells, zero rejected cells, zero reasoning tokens, and
  `$0.06605485` cost.
- Boundary projection yielded 2 or 3 distinct physical LLM root endpoints; no action
  was padded or inserted.

## Primary Pilot Result

Positive entropy-AUC gains favor LLM d2.

| Comparison | Mean gain (nats) | 95% paired bootstrap CI | W / T / L |
| --- | ---: | --- | --- |
| LLM d2 - shared LLM d1 | -0.0402 | [-0.1761, +0.1119] | 3 / 1 / 4 |
| LLM d2 - LLM matched width | +0.1506 | [-0.1153, +0.4401] | 3 / 0 / 5 |
| Grid d2 - grid d1 | +0.0587 | [-0.0683, +0.2504] | 2 / 1 / 5 |
| LLM d2 - grid d2 | -0.0191 | [-0.2970, +0.2262] | 4 / 0 / 4 |

The d2-versus-d1 true-particle log-posterior AUC contrast was `+0.0422` nats but had
CI `[-0.1450, +0.2549]`. The depth-by-proposal interaction was `-0.0989` nats with
CI `[-0.3072, +0.1001]`.

## Decision

The pilot fails its preregistered promotion rule and does **not** justify a 30-trial
confirmation. The task/interface is still tractable and mechanically sound, but this
estimator used four stochastic outer branches and Monte Carlo child-action rankings.
The separately preregistered quadrature rung changes only that avoidable numerical
variance before retesting a fresh paired pilot.
