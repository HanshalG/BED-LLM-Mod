# Bongard Confirmation Account-Wide Budget Chain Correction

Date frozen: 2026-08-09, before any confirmation response and before the
development outcome that can authorize confirmation.

The unopened August 15--18 confirmation launcher previously opened each daily
ledger at its process-start cumulative OpenRouter usage. That could omit
unrelated account spend earlier on the same London calendar day.

Prospectively, August 15 inherits the independently linked August 14 paired
Development64-plus-naive close. August 16--18 inherit the preceding verified
confirmation ledger close. The inherited boundary is the predecessor ledger's
`opening_total_usage_usd + recorded_actual_spend_usd`. All live cumulative usage
above that boundary counts against the new $5.00 account-wide cap. Before any
ledger or paid call, the launcher requires this preexisting spend plus the full
$4.75 confirmation block cap to be at most $5.00. Negative usage deltas,
malformed predecessors, insufficient allowance, and non-finite account values
fail closed without creating a ledger or invoking the model.

Delayed posting after a predecessor close is conservatively charged to the next
day. Reconciliation remains the maximum of posted usage since the inherited
boundary and locally measured accepted-request cost. Unspent allowance never
rolls over.

This accounting-only correction changes no task, image, prompt, schema, seed,
policy, action, endpoint, control, threshold, model, request count, or analysis.
