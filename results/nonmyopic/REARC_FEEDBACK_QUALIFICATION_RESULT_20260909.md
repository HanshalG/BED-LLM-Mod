# Feedback qualification: initial coverage null

Frozen implementation285c6885; completed all16 initial proposal/repair calls once.
Only1/8tasks has a program matching its first visible demonstration, below the
frozen6/8gate. No paired aware/blind refresh, symbolic comparison, forecasts,
target labels or depth runs opened. All eight source collections succeeded.

| Task | Proposal batch | Repair batch | Matching programs, proposal / repair |
|---|---|---|---|
|4be741c5|4 wrong grids|invalid references|0 / 0|
|b7249182|invalid step IDs|invalid references|0 / 0|
|c1d99e64|invalid references|3 matches,1 execution error|0 / 3|
|963e52fc|4 wrong grids|invalid references|0 / 0|
|10fcaaa3|3 wrong grids,1 invalid output|2 wrong grids,2 execution errors|0 / 0|
|f35d900a|4 wrong grids|2 wrong grids,1 invalid output,1 execution error|0 / 0|
|a68b268e|4 wrong grids|invalid references|0 / 0|
|c9f8e694|4 wrong grids|invalid output reference|0 / 0|

Seven of16entire batches failed graph validation (2proposal,5repair). The strict
batch rule was frozen; no partial salvage was performed or counted as success.
Among the9accepted batches,31/36programs returned grids but only3matched the visible
example, all on one task. These are dependent candidate counts, not36independent
trials. Feedback helps one task but is not reliably sufficient on this cohort.

All16calls completed with finish_reason stop; reasoning tokens ranged516..3997.
Cost $0.07757344; uncertain exposure0. This is not evidence of a thinking-budget
ceiling. The failure combines symbolic-reference brittleness and wrong executable
transformations; merely fixing parser errors would not demonstrate correctness.

Exact offline replay verifies all16requests/receipts, diagnostic caches, eight
update records and the identical initial-coverage result with zero new calls.
Original implementation/artifact hashes verify. This is a qualification null,
not a negative estimate of non-myopic policy value: no policy endpoint opened.
Do not reduce the coverage threshold, select the one successful task for a fresh
claim, rerun these seeds or launch deeper planning from this pool.

Next research decision: a source-independent proposal language or model gate
must establish reliable executable semantic coverage before more nested planning.
A more constrained graph schema can address references, but cannot by itself
repair the23wrong proposal grids. Any descendant requires a new prospective
interface/cohort and matched controls; this report authorizes no immediate spend.

53tests passed before launch; exact replay passed after completion. Closing
credits/usage/balance245/221.390706779/23.609293221; conservative London-day spend
$0.9724283, remaining$4.0275717. No cluster or automation changes.
