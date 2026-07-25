# Zendo Particle-Multiset Confirmation Result

Date: 2026-07-25

Status: **failed closed before branch generation, scoring, or endpoints**.

Run: `zendo-particle-multiset-confirmation-20260725T073814Z`

Preregistration:
`results/nonmyopic/ZENDO_PARTICLE_MULTISET_CONFIRMATION_PREREGISTRATION.md`

## Outcome

All seven initial GPT-5.4 requests completed, but parsing stopped on the third
task, `kappa`. Hypothesis H02 used an attribute predicate with
`attribute="color"` and `value="large"`. Both strings individually occur in the
DSL, but that cross-field pair is invalid: `large` is a size, not a color.

The strict executable validator rejected the population. No branch-refresh or
scorer request was made, and no hidden task predicate, root endpoint, policy
selection, or aggregate gate was evaluated. There was no field normalization,
AST repair, response replacement, reissue, or partial-task continuation.

The first two stored initial populations, `upsilon` and `iota`, each had 12 valid
unique ASTs. Four later batch responses were already served but were not copied
into the checkpoint because the parsing loop stopped at `kappa`; they are not
used diagnostically or scientifically.

## Serving diagnostics

- Physical requests / HTTP attempts: `7 / 7`
- Prompt / completion / reasoning tokens: `3,140 / 4,459 / 0`
- Retries / forced exits: `0 / 0`
- Cost: `$0.074735`
- Stored valid initial populations: `2`
- Stored invalid initial populations: `1`
- Branch requests: `0`
- Scorer requests: `0`
- Hidden endpoints: `0`

This exact free-form-AST confirmation interface is closed. The result is a
serving/schema null, not evidence for or against cross-rule non-myopic efficacy.
The failure also exposes a concrete engineering limitation: prompt-described
cross-field type constraints are not reliable enough for an 84-call
fail-closed executable-belief experiment.

## Reproducibility

- Public failure:
  `results/nonmyopic/zendo_particle_multiset_confirmation/RESULT.json`
- Private raw checkpoint SHA-256:
  `d93af106439e3c8292b5a01fc062aa66fdcbb90a181e7dbabc9703a3735e7825`
- Run failure artifact SHA-256:
  `a6b8f1a904d845918500900fc219a00eb219835010e31da32807f480efaaf7ac`

Raw responses remain untracked.
