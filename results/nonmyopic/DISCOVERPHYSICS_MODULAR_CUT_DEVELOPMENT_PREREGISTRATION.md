# DiscoverPhysics Modular-Cut Development

## Status

Frozen before computing the modular-cut counterfactual. The conservative cap
is closed after missing its two-tree magnitude gate and is not retuned.

## Causal Motivation

Branch-conditioned support is generated after the LLM sees a representative
root observation. Component identity is therefore a data-dependent modeling
choice, not an ordinary latent variable sampled before the data. Feeding the
same trajectory back into a component Bayes factor can promote a misspecified
generated module from 5% prior mass to 25--71% posterior mass.

Modular Bayes addresses feedback between misspecified modules by cutting the
feedback path while preserving inference inside each module.

## Sole Candidate Rule

Compute exact full-history conditional posteriors separately inside the
initial and refreshed supports. Keep the component probability fixed at its
frozen generative prior:

```text
cut_prediction =
    0.95 * initial_component_prediction
    + 0.05 * refresh_component_prediction
```

Unlike the rejected conservative cap, this rule does not retain negative
component-level feedback: refreshed mass is exactly 5%, not
`min(component_posterior, 5%)`. It is fixed by the causal module structure,
not fitted to either endpoint.

There is no grid, temperature, region exception, branch exception, support
edit, or alternate component mass.

## Development Replays

Use the exact two already-open 384-map endpoints:

1. original tree, seeds `24600--24617`; and
2. structured fresh tree, seeds `24700--24717`.

Reproduce uncapped retained B and fixed B below `1e-10` maximum absolute error.
Use each endpoint's frozen common random numbers and stratified bootstrap seed.

## Endpoint Gates

On both trees:

- cut B reduces weighted MSE by at least `1%` versus fixed B; and
- the paired region-stratified 95% interval for
  `MSE(fixed B)-MSE(cut B)` has positive lower bound.

## Internal-Policy Gates

Only if all endpoint gates pass, replay each tree on its own eight-hypothesis
initial support with the frozen internal noise/sample counts. Require:

- immediate EIG selects D;
- modular-cut trajectory risk selects B; and
- B reduces modular-cut internal trajectory risk by at least `10%` versus D.

All gates are conjunctive. Any failure rejects the modular-cut method and
forbids a paid third-tree confirmation. Do not relax the `1%` threshold after
the conservative cap reached `0.855%`.

If every gate passes, separately preregister one new strict-schema support tree
and untouched physical endpoint before any model response.

## Accounting

- New model calls: `0`
- OpenRouter cost: `$0`
- OatML, Slurm, SSH, and cluster use: prohibited
