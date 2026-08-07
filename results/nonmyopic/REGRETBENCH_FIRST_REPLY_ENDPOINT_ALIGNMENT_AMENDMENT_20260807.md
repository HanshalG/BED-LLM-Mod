# RegretBench First-Reply Endpoint Alignment Amendment

Date: 2026-08-07

Status: frozen before any RegretBench policy response or endpoint.

## Problem

The first-reply likelihood-alignment amendment requires at least `40/64`
realized question-one replies per primary policy to occur under a
truth-consistent initial particle. It does not change endpoint values. Thus an
individual realized question-one history that was absent from the planner's
truth-consistent simulated tree can still receive favorable truth mass after
the LLM regenerates support from that unmodelled history.

The existing question-two endpoint already handles the analogous case
conservatively: if the exact realized second reply is absent from the
regenerated likelihood partition, terminal truth mass is zero. The first and
second transitions should follow the same alignment rule.

## Amendment

For each primary policy path, retain the existing action-validity flag: both
questions must be officially supported and their mapped facets must differ.
Add a likelihood-aligned trajectory flag that additionally requires the exact
realized first reply to occur under at least one truth-consistent initial
hypothesis, using the already frozen alias matcher and reply normalization.

If that first-reply condition fails:

- retain raw truth mass after question one, raw aligned terminal truth mass,
  and raw fresh-regeneration truth mass as descriptive diagnostics;
- set scored truth mass after question one to `0`;
- set scored aligned terminal truth mass to `0`, Brier to `1`, and log loss to
  the frozen `1e-12` probability floor; and
- set scored fresh-regeneration truth mass to `0` with the same worst-case
  Brier and log loss.

Action-invalid paths continue to receive the existing identical penalties.
An exact question-two reply absent from the generated likelihood partition
continues to produce terminal mass zero through the existing endpoint formula.
The optional naive-thinking baseline has no initial generated likelihood tree
and remains governed only by its existing descriptive action-validity rule.

The public task row contains the new boolean but no raw question, reply, alias,
facet, or truth identity. The independent verifier reconstructs every raw and
scored field from the private initial support, official mapping, and frozen
truth control.

## Scope

This amendment changes no prompt, model, generated tree, candidate, policy
selection, realized action, seed, cohort, request count, bootstrap, scientific
threshold, or budget. It preserves or worsens every individually unmodelled
path and cannot manufacture truth mass. Because different policies can select
different first questions, paired policy differences may move in either
direction; this endpoint rule is therefore frozen before responses rather
than characterized as globally non-rescuing.

The existing enriched-smoke `3/3` and formal `40/64` mechanics gates remain
mandatory. Development and the already frozen confirmation use this same
endpoint rule.
