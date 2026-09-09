# Retrospective ordinary-horizon oracle diagnostic

Use only the exact bank34ba78da from the opened August15 study:22models,
15pair truths,9actions,28target protocols. No source simulation or LLM calls.
Two explicitly oracle conditions: uniform all22models, and uniform15pair models.
In each, the same posterior drives observation branches, forecasts and squared
prediction risk. No restricted forecast support, oracle proposals, or regeneration.
The realized evaluation averages uniformly over all15original pair truths.

Observations are deterministic integer counts exactly as banked. Replace the old
Gaussian pseudo-update by exact consistency under this explicitly deterministic
model; this is a new descriptive estimand, not a replay of the old ladder.
All9actions always available until used. Four-query physical budget for every arm.
At every step h1/h2/h3 minimize expected terminal28-query MSE after precisely
min(h,remaining) observations, then execute one and replan. Exact rational
arithmetic and smallest-action ties. No policy-level recursion. Include exact
uniform-random-without-replacement four-query control, not a lucky random seed.

Bank every truth's actions, observations, final loss; means and roots for both
conditions. Resource caps60seconds/250000new cached value/random states per
condition; stop and retain failure prefix, no increased cap or omitted worlds.
Tests must cover true horizon stopping, multicategory observations, posterior
prediction, random averaging, tie handling and caps before evaluation.

This is retrospective source understanding. No new positive gate, paid permission,
LLM necessity, calibration, or rescue of the old failed formulation follows from
any outcome. Source and protocol hashes accompany the exclusive output. Old banks,
reports and gates remain unchanged. No new simulator outcomes or endpoint cohort.
