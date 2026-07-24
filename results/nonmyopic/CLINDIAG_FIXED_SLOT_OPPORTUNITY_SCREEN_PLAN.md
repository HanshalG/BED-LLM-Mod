# ClinDiag Fixed-Slot Structural Opportunity Screen

Date: 2026-07-24

Status: **stage 1 preregistered before any model response.**

## Question

Before generating any two-step path, test the necessary condition for a non-myopic
truth-recovery opportunity: does the truth remain absent after every individual stored
evidence action on at least two fresh cases?

This is a truth-anchored mechanism screen, not a deployable policy evaluation. The
truth is hidden from support generation and enters only post-generation semantic
measurement.

## Frozen Cases

Seed `24295` selected two challenging and two rare cases from the fixed-slot eligible
pool, excluding the two serving-smoke cases:

- challenging `20220188`;
- challenging `11222813`;
- rare `rare140`;
- rare `rare122`.

Selection was mechanical and fixed before model calls.

## Stage 1

For each case, GPT-5.4 non-reasoning generates:

1. one 12-diagnosis support from the initial presentation;
2. one refreshed 12-diagnosis support for each of the eight generic fixed actions,
   independently conditioned on the same initial support.

GPT-5.4 Mini non-reasoning then makes one strict semantic truth-equivalence
measurement per case over the nine supports. Raw measurement responses are persisted
even on parser failure.

Frozen serving details:

- support temperature `0.5`;
- measurement temperature `0.0`;
- zero structured retries;
- exactly 40 physical requests;
- OpenRouter run ceiling `$0.50`;
- projected ledger reservation `$0.25`;
- live credits and the stricter project ledger checked immediately before launch.

## Frozen Gate

A case has `two_step_room` only when:

- its initial truth-match score is below `0.80`; and
- its maximum truth-match score over all eight one-step supports is below `0.80`.

All gates must pass:

1. exactly 40 requests;
2. zero reasoning tokens and zero retries;
3. all 36 supports parse to 12 diagnoses;
4. no full hidden-target string occurs in the initial presentation or eight stored
   source observations;
5. at least two of four cases have `two_step_room`;
6. no parser or runtime failure.

If fewer than two cases retain headroom, the coarse eight-slot construction is
structurally saturated and no pair generation follows. If at least two pass, a
separate preregistration will freeze exhaustive ordered two-step generation only for
those mechanically selected cases, including exact-prompt validation of the oracle
and greedy-continuation terminal paths.
