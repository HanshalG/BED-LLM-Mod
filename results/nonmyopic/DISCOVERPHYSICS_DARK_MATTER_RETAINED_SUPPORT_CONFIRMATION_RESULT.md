# DiscoverPhysics Retained-Support Fresh-Map Confirmation

## Decision

**All preregistered gates passed on 384 fresh maps.**

This is the first confirmation in the project where an LLM-generated,
branch-conditioned hypothesis component improves both a non-myopic policy
over its myopic counterpart and the same-root fixed-support control.

## Frozen Policy

The policy was fixed before generating the confirmation endpoint:

- `.95` branch mass on the original eight-hypothesis support;
- `.05` branch mass on the eight LLM-regenerated hypotheses;
- exact full-history likelihood-ratio updates from the official simulator;
- the original frozen branches and continuation actions; and
- no new LLM outputs, calibration, deduplication, or weight search.

The `.05` mass came from a disclosed post-hoc curve on a disjoint, already-
open 96-map development endpoint. That curve did not itself support a claim.

## Root Selection

| Root | Immediate EIG | Retained-support internal risk |
|---|---:|---:|
| A | `1.1862` | `.8005` |
| B | `1.0953` | `.4523` |
| C | `.9049` | `1.6925` |
| D | `1.2177` | `1.6864` |

Myopic selected northeast target D. Retained-support lookahead selected
central scout B, reducing internal trajectory risk by `73.2%`.

## Fresh Endpoint

The confirmation used seeds `24600--24615`: 384 previously ungenerated maps,
96 per region, with new observation-noise seed `24616`.

| Policy | Official held-out trajectory MSE |
|---|---:|
| Retained-support center B | `2.9284` |
| Retained-support northeast myopic D | `3.7854` |
| Retained-support random northwest A | `3.7634` |
| Fixed-support center B | `3.0006` |

Retained-support B improved over:

- myopic D by `22.6%`, with paired 95% interval
  `MSE(D)-MSE(B) = [.6393, 1.0878]`;
- random A by `22.2%`; and
- same-root fixed-support B by `2.41%`, with paired 95% interval
  `MSE(fixed)-MSE(retained) = [.0341, .1100]`.

Nearest-support trajectory risk improved from `1.7745` to `1.3816`, a
`22.1%` reduction. Every frozen selection, effect-size, uncertainty,
random-control, fixed-support, and coverage gate passed.

## Regional Check

This decomposition is descriptive:

| Region | Retained B | Fixed B | Fixed minus retained |
|---|---:|---:|---:|
| NE | `2.5324` | `2.8269` | `.2945` |
| NW | `1.7626` | `1.5210` | `-.2416` |
| SW | `5.8277` | `5.8293` | `.0017` |
| SE | `2.2111` | `2.4768` | `.2657` |

Retained support wins 221 of 384 paired maps. The asymmetric prior gives NE
the largest weight, so NE and SE gains outweigh the NW loss while SW is at
parity.

## Interpretation

The central scout has lower immediate EIG than the northeast target but
enables branch-specific continuation actions and LLM-generated hypotheses.
Exact likelihood verification recovers the non-myopic root. Crucially, the
fresh-map gain over same-root fixed support isolates a small benefit from
path-dependent hypothesis generation rather than from choosing the central
root alone.

The result is LLM-native in support construction, not in likelihood
estimation: the LLM generated executable open-space source configurations
after hypothetical branch histories, while the official N-body simulator
computed all probabilities and endpoints.

Limitations remain substantial:

- only one frozen GPT-5.4 support tree is confirmed;
- the `.05` mixture mass was selected on a prior development endpoint;
- the fresh endpoint changes physical maps and noise, not LLM generation;
- the gain over fixed support is statistically clear but small; and
- this does not validate prompt-only LLM likelihoods or scoring.

## Accounting

- New model/API calls: `0`
- OpenRouter cost: `$0`
- OatML use: none
- Confirmation result JSON:
  `results/nonmyopic/discoverphysics_dark_matter_retained_support_confirmation.json`
- Confirmation JSON SHA-256:
  `aecc11fc2b5efc1e7755137d727a58a7fd69d107ca19035b55d38015145050b9`
- Source policy SHA-256:
  `ab2b8a4ce3134fd12236e316b126cf00931f15cb78f818d00cca0b3a109abb46`
- Source model-state SHA-256:
  `7b13e0a3d105b3c823efe3f7bfe66dfb32f983058c9e0a279cd8f366c5a4b8e0`
- Development curve SHA-256:
  `b8aed4f44e72def92383b2f3171654e03ecf26b8c125ba26f98158308f430fcd`
