# Scene facts: prospective observation-interface qualification

## Evidence and intervention

The closed native-revision study fitted both observations on only 1/4 tasks.
Saved public code/feedback distinguishes inference from implementation failures:
task 88a10436 repair slot 0 treats every color-9 cell as a marker, although the
first observed input has 578 color-9 cells out of 621 and the second has only
3 out of 275. Its first-output mismatch is 542 cells and second-output mismatch
is zero. The plan said rare markers, but the implementation omitted that guard.
Task d4f3cd78 repair slot 0 fits the first observation and misses three cells on
the second. These are observations, not proofs that a particular fix will work.
No future labels are consulted or authorized on that closed cohort.

Test a genuinely new prompt interface, not a stronger model on failed instances:
raw grids versus the same grids with deterministic per-grid scene facts.
Facts include dimensions, color counts, all tied modal colors, same-color
four-connected component counts, singleton counts, bounding boxes and rectangular
occupancy. At most eight components per color are listed, with explicit omitted
counts. No color is declared background, no semantic rule is inferred, and no
reference implementation or unseen output is available to the summarizer.
The treatment includes the fixed role-binding instruction in rearc_scene_update;
it is a composite interface intervention, not an isolated token-count effect.

## Frozen comparison

- Model: exact openai/gpt-5.6-luna, medium reasoning, OpenAI-only, no fallback.
- Four fresh tasks: before responses, freeze a 24-task metadata-selected pool,
  excluding the entire prior reserved-candidate union (98 IDs at this point).
  Reuse the previously tested source-verification rejection rule, never select
  by model performance or inspect future outputs to select tasks.
- Per task: two prescribed observed input/output pairs, two public query inputs,
  seven public terminal inputs. Both arms receive the same two observations
  from the start; this experiment does not measure sequential revision itself.
- Each arm independently makes three calls: four contrasting plans, eight native
  Python programs, one feedback-driven eight-program repair. Six calls per task,
  24 total. Both retain all 16 fixed slots and use identical numerical posterior
  conditioning. Invalid execution mass is retained in predictions.
- Paired arm seeds: 51400 + 3*task_index + stage_index (0,1,2), equal across arms.
  Order alternates raw-first/inventory-first across tasks. Equal seeds do not
  guarantee deterministic provider responses. No retries or extra repair rounds.
- Same 16384 completion limit, 131072 prompt ceiling, 65536 request-byte ceiling,
  and $0.08 worst-case reservation per attempt. Total maximum $1.92, subject to
  the account-wide London-day $5 ceiling and live exact-route pricing. Summary
  prompts can use more input tokens; report actual tokens and cost by arm.

## Prospective decision

Before any future outcome is opened, inventory must yield at least one program
fitting both observations on at least 3/4 tasks. Otherwise bank coverage null;
all 36 future outputs remain sealed. Never use visible fit as generalization.

If coverage passes, seal both complete forecasts before opening those outputs
once. Require jointly: inventory positive exact-answer mass on >=6/8 queries;
nondegenerate query predictions on >=2 tasks; pooled normalized whole-grid Brier
improvement raw-minus-inventory >=0.01 across all 36 outputs; positive paired
task-level Brier improvement on >=2/4 tasks; nonworse pooled fixed-canvas Brier.
Report query and terminal metrics separately, correct-answer mass, failure mass,
and truth-mass versus concentration decomposition. A dispersed wrong pool is
not sufficient. All gates stay frozen after responses; a null closes this cohort.

This small screen cannot establish efficacy, monotonic depth or equivalence.
A pass authorizes only a separately frozen sequential transition-fidelity and
endpoint-blind horizon-opportunity experiment. That experiment must compare
actual versus simulated LLM regeneration, use compute-matched myopic and random
controls, paired outcomes and sealed evaluation. No depth sweep is authorized
by source correctness, schema validity, observed fit or this protocol alone.

## Implementation status

The deterministic summarizer and paired six-call updater are implemented.
15 focused tests pass: ties, connectivity, bounded omissions, color-role reversal,
equal calls/slots/evidence, order reversal, observed-only output summaries,
invalid-history rejection and oversize-message rejection before dispatch.
These tests do not establish model utility. Fresh source/panel, forecast sealing,
budget/receipt integration and exact pass/null replay remain required before
any paid request. No task selection or new model call occurred in this turn.

Authenticated account at 2026-09-09T23:07:53+01:00: credits245,
usage222.126794559, balance22.873205441. Existing London-day ledger validated;
conservative spend1.70851608, remaining3.29148392, prior uncertain .04 retained.
Previous goal turn was a status restatement (no progress). This turn implements
and tests a specific prospective intervention. Full research goal remains unmet.
