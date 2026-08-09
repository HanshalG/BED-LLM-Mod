# Bongard OpenWorld Development Daily Handoff Protocol

Frozen: 2026-08-09 Europe/London, before any August 10 mechanics response,
development response, development endpoint, or confirmation endpoint was opened.

Status: **prospective orchestration-only protocol; conditionally active only
after the existing mechanics authorization**.

## Purpose

Each frozen Development64 day currently uses two paid commands: one 16-task
nonreasoning path-dependent block and its separately labelled Luna-medium naive
baseline. Both already share the original account opening and hard `$5` daily
cap, but manual ordering can omit the baseline after the main block or, on block
D, after the combined development endpoint is visible.

This protocol binds the immutable commands into one daily handoff. It changes no
model, prompt, response schema, task, image, split, seed, reasoning effort,
action, policy, likelihood, endpoint, updater, request count, retry rule, cost
cap, gate, claim tier, confirmation authorization, or headline mapping.

## Required Order

For block `A`, `B`, `C`, or `D`, the handoff must:

1. independently verify the existing August 10 mechanics authorization and all
   prior main and naive blocks through the frozen main preflight;
2. run or validate the frozen main Development16 daily executor;
3. regardless of any observed main endpoint direction, run or validate the
   corresponding frozen Luna-medium naive baseline;
4. bind the main result, main daily execution, main account ledger, naive
   result, naive execution, and naive supplemental ledger in one terminal
   record.

The naive preflight must continue to compute account-wide spend from the main
ledger's opening cumulative usage and the larger of posted and locally recorded
main spend. The combined run-level cap is `$4.95` and remains below the hard
`$5.00` London-day limit.

## Failure And Replay

A banked main failure closes the baseline and the handoff for that block. A
banked naive failure closes the handoff after retaining the main block. Neither
failure authorizes a rerun, replacement baseline, later block, confirmation, or
claim. An unexpected interruption may resume only through each immutable child
executor's existing no-repeat validation path.

An existing terminal handoff must independently revalidate all six bound files
before acceptance. The handoff itself makes no model call, creates no paid-call
authorization, changes no claim tier, and creates no confirmation authority.

## Interpretation

This is selection-bias and operational-completeness protection. It does not make
the naive baseline compute matched, does not strengthen a scientific endpoint,
and cannot rescue or veto the frozen development result. Development remains
conditional on an independently verified mechanics pass.
