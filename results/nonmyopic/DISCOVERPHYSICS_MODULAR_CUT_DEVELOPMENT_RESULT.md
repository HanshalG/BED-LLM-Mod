# DiscoverPhysics Modular-Cut Development Result

## Status

`ENDPOINT_GATE_FAILED`

Modular cut improved same-root fixed B on both open trees with positive paired
intervals. The original-tree improvement was `0.9595%`, below the frozen
`1%` magnitude threshold. The endpoint conjunction fails; internal-policy
gates were not run and no paid third tree is authorized.

## Frozen Rule

Exact full-history inference remains inside each support. Component-level
feedback is cut:

```text
cut_prediction =
    0.95 * initial_component_prediction
    + 0.05 * refresh_component_prediction
```

The 5% refreshed mass is fixed by the causal module prior. No grid,
temperature, endpoint fitting, region exception, or alternate mass was used.

## Results

Both retained and fixed controls reproduce with maximum absolute error at most
`3.6e-15`.

| Development tree | Fixed B MSE | Cut B MSE | Reduction | Paired fixed-minus-cut CI | Gate |
|---|---:|---:|---:|---:|---|
| Original positive tree | `3.000591` | `2.971801` | `0.9595%` | `[0.02489, 0.03255]` | **fail**: below `1%` |
| Structured fresh tree | `3.070884` | `3.024578` | `1.5079%` | `[0.04132, 0.05149]` | pass |

Relative to the rejected conservative cap, modular cut improves both
development endpoints slightly:

- original: `0.855%` to `0.9595%`;
- structured: `1.434%` to `1.5079%`.

The fixed component probability also improves every region on the structured
tree. On the original tree, SW remains slightly worse than fixed support while
the weighted aggregate improves.

## Decision

Do not round `0.9595%` to `1%`, relax the magnitude gate, run the internal
policy stage, or purchase a third-tree confirmation. The rule is rejected by
its frozen conjunction.

The replicated positive direction remains useful mechanism evidence: both
independent trees benefit from cutting feedback into the data-dependent support
module. The remaining weakness is the support generator itself. In the failed
fresh tree, one center branch gave `95.2%` generated mass to NE and only `3.8%`
to NW. A genuinely distinct successor may prospectively require broad regional
coverage and bounded regional mass in every branch before applying modular
cut. That changes the generated hypothesis process rather than retuning the
closed component rule.

## Accounting

- New model calls: `0`
- OpenRouter cost: `$0`
- Internal-policy stage: not accessed
- OatML, Slurm, SSH, or cluster use: `0`
