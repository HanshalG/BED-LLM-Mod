# DiscoverPhysics Robust Step Selection Result

## Outcome

**Development null; global uncertainty-clipped coefficient route closed.**

The frozen maximin selector evaluated the exact grid
`beta = 0, 0.025, ..., 0.500` on all three open endpoints. Fixed-support errors
reproduced within `3.6e-15`.

The selected coefficient was `beta = 0.125`:

| Tree | Fixed B MSE | Selected B MSE | Relative gain | Paired fixed-minus-selected 95% interval |
|---|---:|---:|---:|---:|
| Original full tree | 3.000591 | 2.983213 | 0.579% | [0.01209, 0.02285] |
| Structured-V3 full tree | 3.070884 | 3.023792 | 1.534% | [0.04099, 0.05318] |
| Fixed initial, fresh branches | 3.048537 | 3.038439 | 0.331% | [0.00354, 0.01676] |

Every selected endpoint gain and paired lower bound was positive. One
independently generated full tree exceeded `1%`.

Two preregistered gates nevertheless failed:

- worst-tree gain was `0.331%`, below `0.5%`; and
- arithmetic mean gain was `0.815%`, below `1%`.

No fresh confirmation is authorized. The grid, tree set, or thresholds are not
refined.

## Curve Interpretation

The fixed-initial fresh-branch endpoint is the binding case. Its gain rises
from `0.203%` at `beta=.05` to a maximum near `0.331%` at `beta=.125`, then
declines; its paired lower bound ceases to be positive at `beta=.175`.

The structured tree, by contrast, continues improving through much larger
steps, reaching about `3.54%` at `beta=.5`. The original tree peaks near
`0.79%` around `beta=.25`.

Therefore no single global step can turn the same clipped generated-support
direction into a practically sized, stable effect across branch generations.
The failure is heterogeneous support quality, not just a globally
underweighted update.

This closes the global coefficient-calibration family. A successor would need
an endpoint-blind, event-level quality signal that transfers to a new branch
sample, or a different LLM-native environment.

## Provenance

- Frozen protocol/code commit: `a9fa941`
- Public analysis SHA-256:
  `2755e696e0b85530e99f9b3f2f067a59408aa9d1ba6bb4adf6d39599681e2698`
- New model calls: `0`
- OpenRouter cost: `$0`
- OatML/cluster use: none
