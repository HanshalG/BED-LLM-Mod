# Number Game Deferred Stress Authorization Amendment

Date frozen: 2026-08-06, before the August 7 control, either reliability gate,
or any stress request.

## Motivation

The original control wrapper creates the stress-3584 authorization only when
the complete worst-case tail (`$0.20` reliability plus `$1.55` stress) fits
immediately after control reconciliation. This guarantees stress whenever the
control costs at most `$3.25`, but can omit it when the control is only a few
cents higher even if the two reliability gates cost much less than their
combined caps and the reconciled post-reliability ledger still has at least
`$1.55` available.

The closest matched Qwen control cost already observed is `$3.25718784`.
Permanently skipping the scale-reliability instrument at that boundary would
be a conservative reservation artifact, not a scientific gate.

## Frozen Deferred Rule

Keep the pre-control authorization path unchanged. If it did not create a
stress entry, the August 7 orchestrator may reconsider stress exactly once
after both fixed reliability-128 entries are terminal and independently
verified.

The deferred decision uses only:

- the exact two preregistered reliability entry identities and terminal
  statuses;
- the account-wide recorded spend after both reliability reconciliations;
- the exact `$5.00` Europe/London daily cap; and
- the unchanged `$1.55` stress cap.

It must not inspect model eligibility, support counts, parse counts, selected
model, Brier values, roots, policies, targets, or any other scientific value.
If the reconciled remaining allowance is at least `$1.55`, append the exact
waiting stress entry and record that its stage was
`post_reliability_reconciled`. Otherwise append no entry and stop the stress
line for that day.

The stress runner remains independently fail-closed. Before constructing an
adapter it requires both reliability artifacts, selects a model only through
the frozen mechanics ordering, reads live OpenRouter usage, and applies the
same account-wide `$5.00` budget check. Provider lag or unrelated spend can
therefore still close the call after deferred authorization.

This amendment changes no model, prompt, seed, case, retry, parser, mechanics
gate, selection rule, request cap, or scientific endpoint. Stress interface
advances to `number-game-budget-model-stress3584-2`; reliability interfaces
and all 3,584 stress cases remain unchanged.
