# DiscoverPhysics Dark-Matter Mixture Curve

## Status

**Post-hoc development analysis only. This reused the already-open 96-map
endpoint and authorizes no claim.**

The purpose was to decide whether regenerated hypotheses are uniformly
harmful or whether the preregistered `.5/.5` retained-support mixture simply
gave them too much mass.

## Analysis

The frozen initial and branch-regenerated supports, exact full-history
likelihood correction, center root B, actions, observations, and common
random numbers were unchanged. Refresh component mass was swept from `0` to
`1` in increments of `.05`:

| Refresh mass | Trajectory MSE | Change versus fixed |
|---:|---:|---:|
| `0` | `2.9672` | `0.00%` |
| `.05` | `2.9189` | `+1.63%` |
| `.10` | `2.9437` | `+.79%` |
| `.15` | `2.9660` | `+.04%` |
| `.20` | `2.9858` | `-.63%` |
| `.50` | `3.0777` | `-3.72%` |
| `1` | `3.2490` | `-9.50%` |

The curve has a narrow interior optimum at refresh mass `.05`. Larger
refresh mass becomes steadily harmful after `.15`.

At `.05`, the retained policy wins 58 of 96 maps. Its weighted paired gain
over fixed support is `.04836` MSE, but a descriptive stratified-bootstrap
95% interval is `[-.0305, .1268]`. The open endpoint therefore does not
establish a positive effect.

## Consequence

The shape supports a precise fresh hypothesis: regenerated support can add
small complementary value when treated as a low-mass expansion rather than
a replacement or equal expert.

The `.05` refresh mass is now fixed. A 384-map confirmation with new map,
noise, and bootstrap seeds has approximate paired power to distinguish the
development effect from zero. It must be preregistered before any new map is
generated and must require a positive paired lower bound. No other weight
may be evaluated on that endpoint.

## Accounting

- New model/API calls: `0`
- OpenRouter cost: `$0`
- OatML use: none
- Result JSON:
  `results/nonmyopic/discoverphysics_dark_matter_mixture_curve.json`
- Result JSON SHA-256:
  `b8aed4f44e72def92383b2f3171654e03ecf26b8c125ba26f98158308f430fcd`
- Source policy SHA-256:
  `ab2b8a4ce3134fd12236e316b126cf00931f15cb78f818d00cca0b3a109abb46`
- Source frozen-model SHA-256:
  `7b13e0a3d105b3c823efe3f7bfe66dfb32f983058c9e0a279cd8f366c5a4b8e0`
