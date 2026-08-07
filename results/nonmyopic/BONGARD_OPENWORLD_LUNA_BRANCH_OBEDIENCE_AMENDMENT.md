# Bongard Simulated-Branch Obedience Amendment

Date frozen: 2026-08-07, before any Bongard model response or scientific
endpoint.

## Gap

The existing nontrivial branch-sensitivity gate intentionally ignores the
newly labelled query image. It requires the positive and negative branch calls
to change predictions on still-unobserved images or to generate different rule
sets. This prevents a model from passing path dependence merely by copying the
simulated label.

That gate does not establish the converse requirement: the regenerated belief
must actually respect the simulated label. A model can generate different
rules for the two prompts while assigning high positive probability to the
query image under both the positive and negative branches. Such a branch state
is not a valid posterior for planning even though it is schema-valid and
support-sensitive.

## Frozen Gate

For every answer-conditioned branch call, evaluate the regenerated mixture's
truth probability on the newly labelled query image using its emitted
history-conditioned weights and image likelihoods. Report branch-label Brier
separately for simulated positive and simulated negative calls.

Each serving, mechanics, development-block, and confirmation-block gate now
requires both class-conditional mean Brier scores to be strictly below `0.25`,
the constant-half Brier score. Every branch belief must contain its exact
simulated label in the supplied history. The check is mechanics-only and opens
no actual candidate or endpoint label.

Matched history-blind branches are excluded from this gate because their prompt
intentionally omits the simulated answer; they continue to receive the same
analytical update used by the frozen control.

## Invariants

This amendment adds no request and changes no image, prompt, response schema,
hypothesis count, policy, score, seed, task, date, budget, endpoint, or
scientific threshold. It supplements rather than replaces the still-unobserved
branch-sensitivity gate.

Interface versions advance to serving `-3`, mechanics `-7`, development `-8`,
and confirmation execution `-3`. Development and confirmation manifests must
be rebound before any response.
