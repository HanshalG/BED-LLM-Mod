# DiscoverPhysics Uncertainty-Clipped Blend Development Result

## Outcome

**Development null under the frozen magnitude gates, with consistent
directional calibration evidence.**

The sole preregistered one-RMS clipping rule reproduced fixed-support errors to
`3.6e-15` and improved over fixed B on all three already-open endpoints. Every
paired lower bound was positive:

| Tree | Fixed B MSE | Clipped B MSE | Relative gain | Paired fixed-minus-clipped 95% interval |
|---|---:|---:|---:|---:|
| Original full tree | 3.000591 | 2.992324 | 0.276% | [0.00615, 0.01044] |
| Structured-V3 full tree | 3.070884 | 3.050453 | 0.665% | [0.01796, 0.02290] |
| Fixed initial, fresh branches | 3.048537 | 3.042343 | 0.203% | [0.00355, 0.00887] |

The arithmetic mean relative gain was `0.381%`.

Two frozen gates failed:

- neither independently generated full tree reached `1%`; and
- the mean gain did not reach `0.5%`.

Therefore this exact 5%, one-RMS rule is closed and does not authorize a fresh
confirmation. The positive direction and intervals must not be substituted for
the preregistered magnitude criteria.

## Mechanism

The mean clip factors were:

- original tree: `0.446`;
- structured tree: `0.573`; and
- fixed-initial fresh branches: `0.369`.

The effective generated-support step was therefore only about
`1.85%--2.87%`, rather than the nominal `5%`. This explains both sides of the
result: clipping removes enough extreme generated displacement to turn the
fresh-branch null positive, but it also attenuates the useful support correction
until the effect misses the minimum practical magnitude.

This is the first calibration rule in this sequence whose support effect is
positive with a positive paired interval on all three open endpoints. It is
development evidence, not a confirmation result, and the three endpoints are
not treated as independent because two share the same initial support.

## Provenance

- Frozen code/protocol commit: `9e911ff`
- Public analysis SHA-256:
  `f4a31f3d5087671180d1eb56c9b8dd65ec179127a72f8fc2cf2646dc80c7975a`
- New model calls: `0`
- OpenRouter cost: `$0`
- OatML/cluster use: none
