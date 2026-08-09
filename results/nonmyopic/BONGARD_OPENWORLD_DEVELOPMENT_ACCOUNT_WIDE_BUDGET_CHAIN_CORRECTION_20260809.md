# Bongard Development Account-Wide Budget Chain Correction

Date frozen: 2026-08-09, before any Development64 or confirmation response.

## Defect

The unopened August 11--14 Development64 daily launcher initialized each new
ledger from the live cumulative OpenRouter usage observed when that launcher
started. This could erase unrelated account spend earlier on the same London
calendar day. The main launcher also reserved only its own $4.75 cap even though
the preregistered Luna-medium naive first-link baseline is a mandatory same-day
$0.20 component.

## Prospective correction

Each development day inherits an immutable cumulative-usage boundary from the
previous independently reconciled close:

- August 11 inherits the exact August 10 Bongard ledger close.
- August 12--14 inherit the preceding day's completed paired
  Development64-plus-naive supplemental ledger, cryptographically linked by the
  paired daily handoff result.

The new day's preexisting account spend is
`live cumulative usage - inherited boundary`. It is nonnegative and counts
against the $5.00 account-wide London-day cap. Before the main ledger is written,
the launcher must prove that this spend plus the full $4.95 paired exposure is
at most $5.00. It then rereads authenticated live usage immediately before
creating the ledger and repeats that proof. A negative delta, malformed prior
close, insufficient allowance, or intervening usage race creates no ledger,
component, or paid call.

The paired naive launcher retains the main ledger's inherited boundary. It
rereads live usage immediately before creating its supplemental ledger and
reauthorizes its full $0.20 exposure against the account-wide spend already
recorded by the main component and any newly posted or unrelated usage.

Unspent allowance never rolls over. Delayed posting after a predecessor close is
conservatively charged to the next day. Reconciliation remains the maximum of
posted usage since the inherited boundary and locally measured accepted-request
cost.

## Scientific invariance

This correction changes no task, image, prompt, response schema, seed, policy,
action, endpoint, control, threshold, model, reasoning setting, request count,
or analysis. It can only refuse execution earlier when the account-wide budget
is unavailable.
