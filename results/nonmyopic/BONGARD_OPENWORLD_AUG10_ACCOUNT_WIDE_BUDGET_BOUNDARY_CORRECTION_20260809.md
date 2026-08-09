# Bongard August 10 Account-Wide Budget Boundary Correction

Frozen: 2026-08-09 Europe/London, before any August 10 Bongard model
response, candidate label, endpoint label, or terminal artifact was opened.

Status: **prospective execution-safety correction; changes no model request,
scientific instrument, component cap, or authorization rule**.

## Defect

The August 10 wrapper correctly reconciled cumulative provider usage after its
daily ledger existed, but initialized `opening_total_usage_usd` from the live
snapshot immediately before Bongard execution. Any unrelated OpenRouter spend
earlier on August 10 would therefore disappear behind a newly opened ledger,
contradicting the registered account-wide `$5.00` London-day cap.

## Frozen Boundary

An authenticated zero-call OpenRouter read on August 9 recorded:

- cumulative credits: `$245.000000000`;
- cumulative usage: `$220.121013787`;
- balance: `$24.878986213`.

The August 10 ledger must use `$220.121013787` as its immutable
`opening_total_usage_usd`, regardless of the live usage when execution starts.
This is deliberately conservative: any account spend after this amendment but
before London midnight is also charged against August 10.

Before writing the ledger or opening a model component, the same-day preflight
must require:

```text
(live cumulative usage - 220.121013787) + 2.00 <= 5.00
```

where `$2.00` is the complete frozen serving-plus-mechanics cap. A negative
usage delta, non-finite account value, or insufficient remaining allowance
fails before any execution artifact or paid request.

## Replay And Reconciliation

The ledger records the frozen boundary, the live pre-execution snapshot, and
the already-spent delta. Existing per-component authorization and post-call
reconciliation remain unchanged and continue to record the maximum of provider
posted usage since the frozen boundary and locally accepted-request cost.
Restarts reuse that ledger; they cannot move the boundary or repeat a banked
component.

This correction supersedes only the old paid-wrapper implementation hash and
the downstream hashes that bind it. All prompts, images, tasks, seeds,
response schemas, retry limits, request counts, scientific gates, endpoint
sealing, `$0.25` serving cap, `$1.75` mechanics cap, and result interpretation
remain unchanged.
