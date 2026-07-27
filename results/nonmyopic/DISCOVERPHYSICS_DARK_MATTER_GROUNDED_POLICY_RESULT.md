# DiscoverPhysics Dark-Matter Simulator-Grounded Policy Result

## Decision

**The policy passed every root, endpoint, uncertainty, random-control, and
support-coverage gate, but failed the fixed-support control. The exact
preregistered result is therefore a null.**

This is the strongest LLM-native directional result in the project, but it
does not establish that branch-conditioned support regeneration adds value.

## Root Selection

Official likelihoods over the fresh LLM-generated executable support gave:

| Root | Probe | Immediate EIG | Internal two-step trajectory risk |
|---|---|---:|---:|
| A | northwest target | `1.1862` | `4.8452` |
| B | center scout | `1.0953` | `1.7450` |
| C | southwest target | `.9049` | `2.2873` |
| D | northeast target | `1.2177` | `4.3916` |

Myopic selected northeast D and simulator-grounded lookahead selected center
B, exactly matching the independent structural oracle. B reduced internal
generated-world trajectory risk by `60.3%` relative to D.

## Fresh Hidden Maps

The model outputs and policies were hashed before the fresh 96-map endpoint.

| Policy | Official held-out trajectory MSE |
|---|---:|
| Dynamic center lookahead B | `3.6161` |
| Dynamic northeast myopic D | `4.4786` |
| Dynamic random northwest A | `5.1932` |
| Fixed-support center B | `2.9672` |

Dynamic center improved over:

- dynamic myopic by `19.3%`;
- random root by `30.4%`; and
- neither fixed support nor the frozen gate: it was `21.9%` worse than
  fixed-support center.

The stratified-bootstrap 95% interval for paired per-map
`MSE(myopic)-MSE(lookahead)` was `[.281,1.427]`, strictly positive.

## Support Mechanism

- All 8/8 refreshed supports differed from the initial support.
- All 4/4 roots had branch-distinct supports.
- Center branches chose southwest and northeast continuations.
- Every refresh retained at least two regions.
- Nearest-support held-out trajectory risk improved from `1.7311` initially
  to `1.4817` after center-conditioned refresh, a `14.4%` gain.

Thus, support regeneration expanded behavioral coverage and the full dynamic
policy beat myopic. However, it still lost to retaining the original support.

## Diagnosis

The dynamic policy routes a continuous first observation into one of two
branches. The refreshed support's LLM weights encode only that branch, and
the final posterior uses the exact continuation likelihood. The precise
continuous first observation is not re-evaluated on the new support.

The fixed-support control instead evaluates both exact observations on the
same hypotheses. It therefore retains more root information even though its
support has worse nearest-map coverage. This is a concrete mechanism for the
fixed-support advantage, not serving noise:

- requests/HTTP: `9/9`;
- cost: `$0.11351`;
- retries, reasoning tokens, forced exits: `0/0/0`; and
- every strict schema and mechanics gate passed.

## Consequence

This exact policy remains failed and is not repaired or rerun. A zero-call
full-history replay may be preregistered to test the diagnosis by evaluating
both root and continuation likelihoods on refreshed supports. It cannot
rescue this result. Any later fresh confirmation would require a
scientifically distinct protocol and independent outputs.

Artifacts:

- Preregistration:
  `results/nonmyopic/DISCOVERPHYSICS_DARK_MATTER_GROUNDED_POLICY_PREREGISTRATION.md`
- Public policy:
  `results/nonmyopic/discoverphysics_dark_matter_grounded_policy/discoverphysics-dark-matter-grounded-policy-20260727T020000Z/POLICY.json`
- Public policy SHA-256:
  `ab2b8a4ce3134fd12236e316b126cf00931f15cb78f818d00cca0b3a109abb46`
- Frozen model-state SHA-256:
  `7b13e0a3d105b3c823efe3f7bfe66dfb32f983058c9e0a279cd8f366c5a4b8e0`
- Private raw-response SHA-256:
  `98c07aa89bdf71c5e67f8fa89439c265b055e2af819a4f12541f21a7cdf4c9b6`
- OatML use: none
