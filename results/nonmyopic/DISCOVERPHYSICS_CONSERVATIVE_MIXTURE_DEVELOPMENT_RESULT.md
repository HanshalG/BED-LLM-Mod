# DiscoverPhysics Conservative-Mixture Development Result

## Status

`DEVELOPMENT_GATE_FAILED`

The fixed conservative rule improved same-root fixed B on both already-open
support trees with positive paired intervals. It missed the preregistered
`1%` magnitude gate on the original tree (`0.855%`), so the conjunctive gate
fails and no paid third-tree confirmation is authorized.

## Frozen Rule

After the exact full-history posterior:

```text
safe_refresh_mass = min(refresh_posterior_mass, 0.05)
safe_prediction =
    (1 - safe_refresh_mass) * initial_component_prediction
    + safe_refresh_mass * refresh_component_prediction
```

Within-component likelihoods, hypotheses, branch routing, continuation actions,
predictions, noise, and endpoints are unchanged. No grid, tuned temperature,
region exception, or alternate cap was computed.

## Numerical Repair

The first original-tree execution was inadmissible before scientific
interpretation. Some combined posteriors rounded refreshed mass to exactly one,
so recovering the tiny initial conditional by division produced infinities.
Uncapped B still reproduced, but fixed B did not.

Commit `1dcaa7c` prospectively repaired only component normalization: initial
and refreshed conditional posteriors are now computed independently in stable
log space. The cap and every gate remained frozen. Both endpoints were then
replayed exactly.

## Results

| Development tree | Fixed B MSE | Capped B MSE | Reduction | Paired fixed-minus-capped CI | Gate |
|---|---:|---:|---:|---:|---|
| Original positive tree | `3.000591` | `2.974949` | `0.855%` | `[0.02169, 0.02942]` | **fail**: below `1%` |
| Structured fresh tree | `3.070884` | `3.026841` | `1.434%` | `[0.03898, 0.04931]` | pass |

Maximum absolute reproduction errors:

| Tree | Uncapped retained B | Fixed B |
|---|---:|---:|
| Original | `1.8e-15` | `3.6e-15` |
| Structured | `1.8e-15` | `3.6e-15` |

The structured-tree cap improves all four regions. Its largest practical
effect is to prevent the branch-conditioned component from taking 33--71%
posterior mass when its frozen prior is only 5%.

## Decision

The rule is directionally compelling: it improves both independent model trees,
has positive paired intervals on both, and repairs the structured tree's
`-5.66%` uncapped support effect to `+1.43%`. It nevertheless fails the
prospective conjunction because the original gain is `0.855%`, not at least
`1%`.

Do not round, relax the threshold, inspect alternate caps, or launch a third
tree for this exact method. The result remains useful mechanism evidence that
ordinary component Bayes over-promotes data-dependent regenerated support.
A future distinct method should derive conservative component weighting from a
proper generative treatment of support selection rather than a hard cap.

## Accounting

- New model calls: `0`
- OpenRouter cost: `$0`
- OatML, Slurm, SSH, or cluster use: `0`
