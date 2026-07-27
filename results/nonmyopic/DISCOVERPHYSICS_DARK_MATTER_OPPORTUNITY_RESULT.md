# DiscoverPhysics Dark-Matter Non-Myopic Opportunity Result

## Decision

**The exact zero-call opportunity gate failed. No LLM smoke is authorized
under this symmetric-prior protocol.**

The official N-body simulator produced a large and replicated directional
scout-then-target advantage. However, the preregistration required the exact
myopic root ID to replicate across two independent Monte Carlo seeds. The
uniform four-region prior makes radius-4.5 probes in different quadrants
physically symmetric, and the two estimates selected `r4.5_a1` and
`r4.5_a7`. This single failed condition is not repaired post hoc.

## Results

| Estimate | Myopic root | Depth-two root | Immediate sacrifice | Total EIG gain | Held-out MSE reduction |
|---|---|---|---:|---:|---:|
| Decision | `r4.5_a1` | `center` | `.1165` nats | `.2462` nats | `51.3%` |
| Confirmation | `r4.5_a7` | `center` | `.0948` nats | `.2509` nats | `59.4%` |

The independent confirmation's official posterior-predictive trajectory MSE
was `.50043` after the myopic root and `.20300` after the depth-two root.
Both policies received an adaptive second experiment from the same 25-action
bank. Thus, the difference concerns the enabling value of the first probe,
not one experiment versus two.

## Interpretation

The depth-two policy stably chose the central scout in both estimates. It
gave up immediate information to identify the halo region before placing its
second probe. The myopic policy instead placed its first probe near one halo
region, obtaining more immediate discrimination there but worse expected
coverage of the other regions.

This is strong structural evidence that the one-active-probe dark-matter
apparatus can contain a genuine non-myopic opportunity with an official
behavioral endpoint. It is not a passed preregistered gate, and it says
nothing yet about whether an LLM can generate, update, or rank the required
semantic hypotheses.

## Consequence

This exact hidden-halo family, uniform prior, action bank, noise level, and
stability criterion close without paid calls. A scientifically distinct
successor may use a prospectively asymmetric spatial prior, which removes
the physical root-label symmetry while preserving the scout-then-target
mechanism. It requires a fresh family seed, fresh Monte Carlo seeds, and a
new preregistration before evaluation.

Artifacts:

- Preregistration:
  `results/nonmyopic/DISCOVERPHYSICS_DARK_MATTER_OPPORTUNITY_PREREGISTRATION.md`
- Public result:
  `results/nonmyopic/discoverphysics_dark_matter_opportunity/OPPORTUNITY.json`
- Result SHA-256:
  `a2a4484a12517c0ef8bda38b8b035d0893b72496a4f0e4834c67c0e56432613f`
- Feature artifact SHA-256:
  `130747b3b8bb69987b8db350d08044377cba7a164f287450637ceb773a47fdba`
- LLM requests: `0`
- OpenRouter cost: `$0`
- OatML use: none
