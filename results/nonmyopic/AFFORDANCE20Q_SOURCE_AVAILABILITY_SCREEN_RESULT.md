# Affordance20Q Source Availability Screen

Date: 2026-07-29

Status: **promising LLM-native substrate, but unavailable; no experiment
authorized**.

## Scientific Fit

Affordance20Q is unusually well aligned with the project goal. Each game hides
an object and asks a Questioner to identify one of eight candidate affordances
by asking free-form yes/no questions about physical properties. The published
benchmark contains 1,009 games over 454 objects and 59 affordances.

The paper says every object has a curated property set and every
object-affordance pair has a label. Realized answers are produced by an LLM
Oracle conditioned on the hidden object's property description. This could
make branch-conditioned counterfactual answers and path-dependent beliefs
irreducibly semantic rather than a wrapper around a hand-coded simulator.

## Availability Audit

The paper claims that all code and data are released at the official
repository, but the artifact is not yet present:

- repository: `https://github.com/1171-jpg/Affordance20Q`
- audited commit: `b7b40dbd7aea191f7e55845857798168bfa58532`
- branches: only `main`
- tags: none
- releases: none
- license: none
- repository contents: one 68-byte `Readme.md`
- README SHA-256:
  `394db13e85929cd15c1b232a272bae5c90a26ddc54130b8d067e57e2fd762a0c`

In issue 1, opened by Hugging Face on 2026-06-15, the author replied that the
team would organize and share the code and data on Hugging Face. No dataset
endpoint or subsequent artifact is currently linked from the paper,
repository, releases, tags, or author Hugging Face profile.

## Consequence

No proxy dataset, inferred property table, paid serving smoke, or policy
experiment is authorized. Reopen this lead only when the official artifact
provides, at minimum:

1. the 1,009 game definitions and candidate affordances;
2. the 454 object property descriptions;
3. the complete object-affordance label matrix;
4. enough Oracle or question-answer machinery to evaluate counterfactual
   branches reproducibly; and
5. usable licensing or redistribution terms.

At reopening, begin with a zero-call source/opportunity audit. Before any paid
policy comparison, establish that at least some roots have a reproducible
non-myopic action preference and that the semantic Oracle is stable enough for
paired common-random evaluation.

