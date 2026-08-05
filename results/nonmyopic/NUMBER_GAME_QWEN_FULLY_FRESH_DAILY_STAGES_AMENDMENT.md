# Number Game Qwen Fully Fresh Daily-Stages Amendment

Date frozen: 2026-08-05, before any formal source or control seed is opened.

## Reason

The fully fresh source+control32 protocol was frozen with a `$9.50` combined
cap and same-process execution. The account now has a hard `$5.00` spend cap
per Europe/London calendar day. This amendment changes only execution timing
and tightens cost limits so the original experiment can run without violating
that constraint.

The source hypotheses, targets, validation supports, control prompts, seeds,
estimands, gates, and endpoint calculations remain exactly those in
`NUMBER_GAME_QWEN_FULLY_FRESH_SOURCE_CONTROL32_PREREGISTRATION.md`.

## Amended Execution

### Source Day

- Run only source trees `100000..100031`, target seeds `100100..100131`, and
  the frozen validation schedule.
- Require a fresh Europe/London daily ledger with the full `$5.00` allowance.
- Tighten the source run cap from `$5.25` to `$5.00`; all other source
  mechanics and scientific thresholds are unchanged.
- Write a hash-bound `CONTROL_AUTHORIZATION.json` after source artifacts are
  complete.

The authorization manifest may read only:

1. whether every source mechanics gate passed; and
2. whether dynamic and fixed-support roots differ on at least 20 trees.

It must not read, copy, hash-select, or branch on source myopic-policy gates,
dynamic-versus-fixed efficacy gates, Brier values, confidence intervals, win
counts, or source status.

### Control Day

- Control execution is allowed only on a later Europe/London calendar date
  with a new daily ledger.
- Recompute the structural authorization from the hash-bound source result and
  require exact agreement with `CONTROL_AUTHORIZATION.json` before adapter
  construction.
- If authorization is true, run all 3,072 fresh history-blind requests with
  the existing `$4.25` cap, regardless of every source scientific endpoint.
- If authorization is false, make zero control calls and bank the original
  mechanics/opportunity failure.

The control is due on the first later day on which its daily ledger is opened;
there is no human endpoint-dependent go/no-go decision between stages.

## Composite Decision

After control completion, construct the same composite endpoint and require
the same source mechanics, source depth-three-over-myopic, source
dynamic-over-fixed, control mechanics, and first-link scientific gates.

The amended operational caps are `$5.00 + $4.25 = $9.25`, stricter than the
original `$9.50` composite cap. Exact request count remains `6,752`. A staged
pass, null, or failure has the same scientific meaning as the original
same-process result.

## Failure And Replay

- No source artifact or response may be replaced between days.
- Any source hash change, authorization mismatch, same-day control attempt,
  expired/missing ledger, or inadequate allowance fails before control calls.
- Existing independent zero-call control replay remains mandatory.
- The original balance-gate failure and synthetic dry run remain immutable.

## Budget State

Spend on 2026-08-05 is already `$0.495741692`; only `$4.504258308` remains,
so the formal source stage is forbidden today. The first eligible source
launch is a later day with a fresh `$5.00` ledger.
