# DiscoverPhysics Conservative-Mixture Development

## Status

Frozen before computing the conservative-mixture counterfactual. This is
posthoc method development on two already-open endpoints, not a new
confirmation and not a rescore of either frozen gate.

## Motivation

The original support tree beat same-root fixed B by `2.41%`; the fresh
structured tree lost by `5.66%`. Exact decomposition of the latter found that
the nominal `5%` refreshed component acquired `25.26%` mean final posterior
mass, and acquired more mass when its held-out prediction was worse.

The regenerated component is data-dependent: the LLM creates it after seeing a
representative root observation. Ordinary component-level Bayes can therefore
be overconfident even though the frozen likelihood correction removes the
representative root likelihood from individual hypothesis logits.

## Sole Candidate Rule

Keep all existing within-component likelihood updates, hypothesis weights,
continuation actions, branch routing, and predictions. After the exact
full-history posterior is computed, set:

```text
safe_refresh_mass = min(refresh_posterior_mass, 0.05)
safe_prediction =
    (1 - safe_refresh_mass) * initial_component_prediction
    + safe_refresh_mass * refresh_component_prediction
```

The cap equals the frozen refreshed-component prior. There is no cap grid,
temperature, fitted parameter, endpoint-dependent rule, region exception, or
support edit.

## Development Data

Replay exactly:

1. the original tree on its already-open 384-map confirmation
   (`24600--24617`); and
2. the structured fresh tree on its already-open 384-map endpoint
   (`24700--24717`).

Use each endpoint's frozen root/continuation sample counts, common random
numbers, regional prior, and stratified bootstrap seed. Reproduce uncapped B
and fixed B numerically before interpreting the cap.

## Gates

The rule passes development only if, on both trees:

- maximum absolute reproduction error is below `1e-10` for uncapped B and
  fixed B;
- capped B reduces weighted MSE by at least `1%` versus fixed B; and
- the 95% region-stratified paired interval for
  `MSE(fixed B)-MSE(capped B)` has positive lower bound.

All gates are conjunctive. A failure rejects this exact cap and forbids a paid
third-tree run for it. Do not inspect region subsets, tune the cap, or substitute
an uncapped/conditional rule.

If both trees pass, separately freeze a third-tree confirmation. That future
protocol must also verify on internal support before opening an endpoint that
the capped-risk policy preserves myopic D, lookahead B, and at least `10%`
B-versus-D risk reduction.

## Accounting

- New model calls: `0`
- OpenRouter cost: `$0`
- OatML, Slurm, SSH, and cluster use: prohibited
