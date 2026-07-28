# DiscoverPhysics Uncertainty-Clipped Blend Development

## Status

Frozen before computing this candidate on any open endpoint.

## Motivation

The fixed 95/5 modular cut makes the non-myopic B root robust, but generated
support value varies across trees: positive on the original and structured
trees, and essentially zero on the fixed-initial fresh-branch tree. The newest
regional decomposition shows that generated support can help some semantic
regions while harming others even when breadth and regional mass are fixed.

The prospective correction should respond to predictive calibration available
at decision time, not endpoint region labels or hidden truth.

## Sole Candidate

For each realized B root/continuation history:

1. compute the exact posterior mean held-out trajectory `mu_initial` within the
   fixed initial support;
2. compute its posterior mean squared trajectory dispersion
   `variance_initial`;
3. compute the exact posterior mean `mu_refresh` within the generated branch
   support;
4. let `delta = mu_refresh - mu_initial` and
   `delta_mse = mean(delta ** 2)`;
5. set

```text
clip = min(1, sqrt(variance_initial / max(delta_mse, 1e-15)))
prediction = mu_initial + 0.05 * clip * delta
```

Thus the generated component keeps its frozen 5% modular role, but its
trajectory displacement cannot exceed one posterior RMS uncertainty of the
initial component before the 5% multiplier. This is a standard robust
contamination-style scale rule with no fitted scalar, region label, endpoint
value, grid, or exception.

## Development Data

Replay with exact common random numbers on all three already-open 384-map
endpoints:

1. original tree / confirmation endpoint;
2. structured-V3 fresh tree / endpoint; and
3. fixed-initial fresh balanced branches / endpoint.

The third shares the structured initial support but has independently generated
branches and an independent physical endpoint. Report each tree separately;
do not treat the three as independent replications in a pooled significance
test.

## Gates

All must pass:

- fixed-support per-map errors reproduce each public endpoint within `1e-10`;
- uncertainty-clipped B has lower weighted MSE than fixed B on all three trees;
- the paired `MSE(fixed B)-MSE(clipped B)` bootstrap lower bound is positive on
  all three trees;
- at least one of the two independently generated full trees improves by at
  least `1%`; and
- the arithmetic mean relative improvement across the three trees is at least
  `0.5%`.

Failure closes this exact rule. Do not change the one-RMS threshold, component
mass, variance definition, tree subset, region weighting, or bootstrap seeds.

A full pass authorizes one separately frozen confirmation using:

- the same public structured-V3 initial support and branches;
- one fresh set of eight balanced branch supports;
- this exact clipped modular policy;
- a new untouched 384-map endpoint; and
- the existing D/B/random/fixed/coverage gates, with clipped B required to beat
  fixed B by at least `1%` and a positive paired lower bound.

## Accounting

- New model calls: `0`
- OpenRouter cost: `$0`
- OatML/cluster use: prohibited
