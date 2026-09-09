# Fresh execution-feedback qualification

Freeze before examples/responses. Cohort is the eight IDs in
REARC_FEEDBACK_COHORT_20260909.json, committed b9b61121 before source inspection.
No substitutions. The previous four-task interface remains closed.

Source-only smoke: every selected task at seeds33000..33002, difficulty[0,1],
single attempt, exact generator/verifier agreement, pinned original source/image,
same nonroot/no-network 256MiB/2CPU-second/15wall limits. Emit hashes/shapes only.
Require24/24 valid before qualification. A failure closes this source cohort for
this protocol, not permission to pick another seed or silently remove a task.

Qualification: demonstrations33100..33102 and targets33200..33207, identical source
settings, one attempt and no substitutions. All prompts see3demo inputs, never
target inputs, IDs, reference code, source descriptions or hidden labels. Initial
and blind updates see only demo0output; aware updates see all3outputs. The new
single-grid semantics and generic identity example are fixed in
rearc_feedback_update.py; the full generic DSL remains available.

Each update makes exactly two calls: four proposals, public execution feedback,
then four repaired/alternative proposals. A syntactically invalid entire batch
gets invalid-batch feedback, not partial salvage; repairs are the explicitly
declared second call, not an unrecorded retry. After that no retry is allowed.
Wrong answers/invalid programs retain diagnostic records. If both batches are
invalid, preserve an empty pool and an explicit unit failure forecast.
The execution worker receives only individual visible input grids; feedback
compares only observed labels. Final repair feedback incurs no third call.

Model exactly openai/gpt-5.6-luna, OpenAI provider only, medium reasoning,
max16384completion tokens and complete serialized request <=32768bytes including
schema and repair history. No silent truncation of a repair message. Initial
seeds33300+2*i and33301+2*i; paired aware/blind seeds33400+2*i and33401+2*i.
No hidden reasoning text or provider fallback. Exactly48calls maximum, $.06 full
per-attempt reserve and $2.88 block cap; hard account-wide$5London-day boundary.
Use the existing authenticated pricing ceilings and fresh authorization before
each dispatch, conservative receipt reconciliation and uncertain reservations.

Perform all8initial two-call updates first. At least6/8 must contain an executable
program exactly matching demo0, else stop before refresh and target outputs.
Retain all8cases; this gate is not permission for survivor-only analysis.
Then aware/blind two-call updates per task, alternating their order by index.
Compare pools initial8, initial8+aware8, initial8+blind8, and the existing bounded
first-order symbolic baseline. Condition every pool on the SAME3demo outputs using
unique-canonical-program uniform priors and deterministic likelihood. Empty pools
produce unit failure forecasts, never resets. The blind control has exactly the
same call/token allowance and feedback mechanism for its visible history.

Seal all8targets' predictions for all8tasks and all4arms before target label
release. Reconstruct outputs against their source hashes. Equal task/target weights;
report normalized whole-grid and fixed900cell Brier, complete failure/duplicate
rates, repair validity and observed-fit change, and all costs/work counts.

Pass only if aware-minus-blind improvement is >=.01absolute and >=10%relative in
mean whole-grid Brier, >=4/8tasks improve by >=.01, and aware is nonworse on BOTH
mean scores than initial, blind and symbolic. Exact receipt/source/forecast replay
and complete coverage required. This is a small development qualification, not
a significance claim, posterior calibration certificate or publishable efficacy.

Even a pass authorizes only a separately frozen branch-transition fidelity and
same-objective horizon-opportunity study. Do not jump to depth sweeps. The symbolic
baseline is limited first-order, so beating it does not establish LLM necessity.
If the signal is only more search or repairs, call it that. No endpoint-informed
gate changes. New paid execution requires source smoke and a connected, tested
48-call budget/sealing/controller runner; neither is authorized by import or by
passing unit tests for the two-call update alone.
