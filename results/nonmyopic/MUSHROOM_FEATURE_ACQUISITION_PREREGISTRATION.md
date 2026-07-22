# Mushroom Feature Acquisition Depth Qualification Preregistration

Registered 2026-07-23 after exploratory zero-call mechanism screens and before the
fresh-seed exact qualification. No LLM response has been requested for this task.

## Purpose

Qualify a non-spatial semantic active-feature task before building an LLM strategy
interface. The target is mushroom edibility under the empirical UCI Mushroom catalog.
A zero-information specimen-collection action permanently unlocks detailed specimen
features, creating a genuine value-of-information decision that one-step EIG cannot
recognize.

## Frozen Environment

- Prior: uniform over all 8,124 UCI Mushroom rows.
- Target: binary edible/poisonous class; unknown/not-recommended species are already
  represented as poisonous by the source dataset.
- Initial field observations: cap shape, cap surface, cap color, population pattern,
  and habitat.
- `collect:specimen`: consumes one round, returns no observation, and permanently
  unlocks the remaining 17 bruising, odor, gill, stalk, veil, ring, and spore-print
  features.
- A feature query deterministically reveals its catalog value and cannot be repeated.
- Posterior: exact uniform conditioning over catalog rows consistent with the complete
  history.
- Horizon: 8 rounds.
- Planning arms: exhaustive exact depth one and exhaustive exact depth two.
- Planning objective: minimize expected cumulative target-class entropy after each
  action within the local planning horizon. This is the local form of the registered
  entropy-AUC endpoint and avoids the terminal-utility postponement artifact.

## Frozen Qualification

- 1,000 paired hidden rows sampled without replacement.
- Fresh seed `24123` and 10,000 paired bootstrap replicates.
- The two policies share every hidden row. Observations are deterministic, so no
  additional outcome randomness exists.
- Primary endpoint: mean post-action class entropy over all eight rounds.
- Corroborating endpoint: mean log posterior probability of the true class.
- Secondary: final entropy, final true-class probability, final MAP-class accuracy,
  action traces, setup timing, and exact scorer units.
- No LLM, OpenRouter, or cluster GPU call is permitted in this stage.

The task qualifies only if all legality/pairing/exactness mechanics pass and both
paired 95% bootstrap lower bounds are strictly positive for depth two minus depth one:

1. entropy-AUC gain;
2. truth-log-posterior-AUC gain.

Passing authorizes only a separately preregistered 10-cell non-thinking 26B serving
smoke. It does not authorize a formal LLM endpoint.

## Exploratory Disclosure

Several free screens were used to choose the task definition. Car Evaluation produced
identical d1/d2 policies. Nursery produced only `+0.0130` entropy-AUC nats by round six.
Deferring four individual Mushroom assays made terminal-EIG d2 worse by `-0.0291`.
The one-time specimen design under terminal utility was also worse by `-0.0254`, while
the endpoint-aligned objective produced `+0.1138` (`[+0.0992,+0.1283]`) on 500 sampled
rows. A fixed-root K4 proxy with an odor-aware continuation beat matched random by
`+0.0626` (`[+0.0534,+0.0724]`). These exploratory rows, seeds `24119`-`24122`, and
threshold observations cannot enter the fresh qualification or any later endpoint.
