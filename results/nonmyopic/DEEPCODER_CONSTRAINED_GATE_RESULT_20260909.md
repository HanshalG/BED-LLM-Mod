# Constrained proposal gate: grammar passes, observed-data coverage fails

Frozen implementation/protocol e6a19298; scoped authorization committed at
dd93158c before dispatch. Full artifacts are in
`results/nonmyopic/deepcoder_constrained_gate_20260909/`.

## Result

All16 planned DeepSeek V4 Flash0731 nonreasoning calls completed on the pinned
OpenInference provider. Every response was source-grammar/schema valid, with
clean stop, zero reported reasoning, bounded tokens and cost. No retries.
The terminal result is **proposal_coverage_null**, not a predictive Brier result.

| Arm | Raw / unique proposed programs summed over calls | Cases with a pool fitting both observed examples |
|---|---:|---:|
| Evidence-aware LLM |44 /12|0/8|
| History-blind LLM |57 /36|0/8|
| Bounded symbolic search |40 successful restarts out of64|8/8|

The LLM unique counts are within-call deduplicated, then summed; they are not
global unique counts. Symbolic restarts took65,050 total operation attempts.
These are observed-history compatibility counts, not target accuracy. The
predeclared >=6/8 aware-coverage requirement failed, so target outcomes were
never opened and no loss/control superiority/depth claim is available.

The forecast seal is
371918b6a0f8a7170bcacf161aba6b973049df203ac66e746b9682d255ed04dd.
Independent zero-call replay checked all16 raw responses against the actual
transmitted schema, compiled every unique program, and recomputed its outputs
on the two ALREADY OBSERVED inputs. All compatibility counts match. The aware
requests contained the correct history, blind requests contained none. The sealed
evaluator replayed the same coverage-null with an endpoint loader that raises
if called. All frozen code/protocol hashes still match; outcomes.json is absent.

## What failed, and what did not

Mechanical grammar validity now works on these16 responses. That is only an
interface result. The generated pools do not even cover the already-observed
examples, so they cannot support the intended Bayesian update or planning pass.
This is missing compatible support, not posterior collapse or a noisy horizon
comparison. Symbolic success shows these observed histories are fit-able under
its broader bounded expression search; it does not establish its generalization.

There is also a concrete representation concern. The request serializer sorts
JSON object keys. In the transmitted schema, each continuation object's `next`
property precedes `statement`. All101 raw returned program roots likewise emit
`next` first, recursively producing later statements before the earlier ones.
The model is therefore asked to serialize computations in reverse dependency
order. The ordering is verified; its effect on inference quality is NOT measured.
Do not attribute the entire null to it or claim that reversing the order would
fix the result. Both controls/cases and interface differ from the previous gate,
so there is no paired causal old-versus-new comparison.

Aware responses also repeat candidates heavily: only12 within-call unique
programs from44 raw proposals. More horizon cannot fix zero observed-data support.
Any successor should first address generation order and evidence-consistency,
with a prospectively specified representation and independent tests, rather than
buy a depth grid or reinterpret this null. This exact gate remains closed;
neither a retry nor a new paid interface inherits its consumed authorization.

## Accounting and verification

Accepted block cost: **$0.008799004**, versus authorized cap$0.25. Uncertain
exposure:0. Including the preceding gate, LondonSept8 spend is$0.008868874,
leaving$4.991131126 in that day's allowance. Authenticated cumulative credits/
usage/balance are245/220.385562868/24.614437132; posted and local costs agree.
No pending reservations remain, and authorization is consumed. Unused allowance
does not authorize filler requests or reopen the terminal gate.

38 focused tests passed in0.92s before launch, including the full synthetic
16-call constrained runner, exact approval checks, last-moment account-spend
race, transport uncertainty/no retry, grammar rejection and endpoint sealing.
Post-run replay used only banked raw responses and real histories. All processes
exited; no cluster/automation changes. The full non-myopic research goal remains
unproved and incomplete.
