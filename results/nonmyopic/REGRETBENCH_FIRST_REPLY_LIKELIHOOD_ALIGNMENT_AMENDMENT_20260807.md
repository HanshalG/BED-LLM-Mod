# RegretBench First-Reply Likelihood Alignment Amendment

Date: 2026-08-07

Status: frozen before any RegretBench policy response or endpoint.

## Problem

The dynamic planner simulates question-one observations using each generated
hypothesis's predicted reply, then asks the LLM to regenerate support from that
simulated history. The existing alignment gate required the exact realized
question-two reply to occur in the regenerated support's likelihood partition,
but did not check the first transition.

Consequently, a planner could rank roots using simulated histories that omit
the exact question-one reply produced by the official environment under every
generated hypothesis compatible with the hidden truth. The realized policy
would then regenerate from a history absent from its scored tree. A favorable
endpoint after that unmodelled transition would not establish non-myopic value
over the LLM's predicted path-dependent belief dynamics.

## Amendment

For a selected first question, a truth-consistent first-reply likelihood match
requires one initial generated hypothesis that satisfies both:

1. its final answer matches one of the hidden truth's frozen answer aliases
   under the existing conservative lexical matcher; and
2. its predicted reply to the selected first question exactly matches the
   official environment reply after lowercase alphanumeric normalization.

The enriched exact-10 smoke must pass this check on all three executed first
questions. In the 64-task policy run, every primary policy must pass it on at
least `40/64` realized histories. Unsupported first actions cannot count.

Public task rows record only booleans and match counts. Raw questions, replies,
aliases, facets, and truth identities remain private. The independent verifier
reconstructs the checks from raw initial supports, official mappings, and the
private truth controls. A changed public count, gate, or amendment hash fails
replay.

## Scope

This amendment adds no model call and changes no prompt, generated tree,
candidate, policy selection, realized action, endpoint value, seed, cohort,
bootstrap, scientific threshold, or budget. It can only convert a run with an
unmodelled first transition into a mechanics failure; it cannot rescue or
improve a policy result.

Both development and the already frozen confirmation use the same gate. The
two 64-task cohorts have zero inter-intent answer-alias collisions under the
frozen matcher, so a truth-consistent alias match identifies exactly one
released intent-equivalence class on every task.
