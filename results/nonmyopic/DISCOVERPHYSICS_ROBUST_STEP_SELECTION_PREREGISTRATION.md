# DiscoverPhysics Robust Step Selection

## Status

Frozen before evaluating any coefficient other than the already closed `0.05`
point.

## Motivation

One-RMS clipping made generated-support value positive with positive paired
intervals on all three open endpoints, but the fixed `0.05` step was attenuated
to an effective `0.018--0.029` and missed the frozen magnitude gates.

This development stage selects a stronger step prospectively. It does not
reinterpret the closed 5% rule.

## Fixed Direction

For every root/continuation history, compute the same endpoint-blind clipped
direction frozen in the previous protocol:

```text
direction = min(1, sqrt(variance_initial / max(delta_mse, 1e-15)))
            * (mu_refresh - mu_initial)
prediction(beta) = mu_initial + beta * direction
```

No region label, hidden truth, endpoint value, or learned feature enters the
direction.

## Coefficient Grid And Selection

Evaluate exactly:

```text
beta in {0.000, 0.025, 0.050, ..., 0.500}
```

on the same three open 384-map development endpoints with exact common random
numbers.

For each nonzero coefficient, compute relative MSE reduction versus fixed B and
the preregistered region-stratified paired interval separately on every tree.

Eligible coefficients must have:

- positive relative gain on all three trees; and
- positive paired lower bounds on all three trees.

Among eligible coefficients, select the one maximizing the minimum relative
gain across the three trees. Break exact ties by choosing the smaller
coefficient. This is a maximin transfer criterion, not a pooled significance
test.

## Development Gates

The selected coefficient must satisfy:

- exact fixed-support reproduction within `1e-10` on all trees;
- minimum relative gain across trees at least `0.5%`;
- arithmetic mean relative gain at least `1%`; and
- at least one independently generated full tree gains at least `1%`.

Failure closes this coefficient-selection route. Do not refine the grid,
change its upper bound, remove a tree, change the clipping direction, or use a
mean-optimal coefficient.

## Authorized Confirmation

A full pass authorizes one separately frozen confirmation:

- public structured-V3 initial support and branches;
- one fresh set of eight balanced GPT-5.4 branch supports;
- the selected clipped-step coefficient;
- a new untouched 384-map endpoint;
- immediate D and robust-lookahead B with at least `10%` internal gain before
  endpoint access; and
- B must beat D by `10%`, random by `5%`, fixed B by `1%` with positive paired
  lower bounds, and improve coverage by `5%`.

The confirmation claim remains conditional on the fixed public initial belief.

## Accounting

- Development model calls: `0`
- Development OpenRouter cost: `$0`
- OatML/cluster use: prohibited
