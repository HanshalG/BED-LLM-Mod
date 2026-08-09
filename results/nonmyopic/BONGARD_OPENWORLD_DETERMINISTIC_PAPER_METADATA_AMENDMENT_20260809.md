# Bongard OpenWorld Deterministic Paper Metadata Amendment

Frozen: 2026-08-09 Europe/London, before any August 10 mechanics response,
development response, development endpoint, or confirmation endpoint was opened.

Status: **prospective V6 paper-integrity amendment; zero model calls**.

## Problem

Mandatory wrapper V5 renders the Luna-only fragment in a random temporary
directory and stores the fragment renderer's returned artifact paths verbatim in
the combined metadata. Those paths contain the random directory name. Therefore
two scientifically identical V5 renders have identical TeX and headline bytes
but different metadata bytes and metadata SHA-256 values.

That does not change any scientific number, but it prevents exact artifact replay
and would make a byte-identical terminal handoff impossible.

## V6 Rule

V6 supersedes V5 for every future development and confirmation paper render.
The combined metadata must store only the Luna renderer's path-invariant summary:

- status, stage, and claim tier;
- TeX, headline, and metadata SHA-256 values;
- model-call and cost values.

Temporary or output-directory paths must not appear in combined metadata. All
other mandatory V5 behavior remains unchanged: exact replay of the classical,
compute-matched, random-strategy, and path-mediation artifacts; directionally
complete reporting; classical headline qualification; and six-page limits.

## Scope

This amendment changes no model, prompt, response schema, task, split, seed,
policy, likelihood, endpoint, request, retry, budget, gate, claim tier,
confirmation authorization, headline rule, metric, or interpretation. It
authorizes no paid call, rerun, development stage, confirmation stage, or claim.
