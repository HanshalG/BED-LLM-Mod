# Structure-only proposal interface

Date: 2026-09-08. Interface mechanics only, zero model calls and $0 new cost.

## Implemented connection

`environments/chembench_mopen/structure_proposer.py` connects bounded LLM response
text to the existing executable-law inference machinery. It has no network
transport, retries, model selection, budget authorization or efficacy gate.

- Explicit public seven-variable box, fixed log-parameter prior and observation
  noise define the proposal task. The only task-dependent prompt data are real
  input/observation history records. No source metadata, true mechanisms, future
  outcomes, targets, or arbitrary task text are accepted by this API.
- History-aware and history-blind prompts have identical instructions and public
  context. The blind prompt contains an empty history and no history length; it
  is invariant to all valid histories. Both downstream numerical fitters still
  receive the complete real history.
- Strict JSON returns one to four equation structures with one to eight named
  parameters each. Priors come from the experiment, not response-supplied bounds,
  fitted estimates, probabilities or evidence values.
- All laws must validate before any are returned. No partial survivor-only batch,
  Markdown repair, retry or silent replacement is provided. Canonical duplicates
  return once and cannot increase prior mass.
- Existing expression/name/AST limits apply; raw responses are capped at 32 KiB,
  histories at 128 records. Literal constants are restricted to 0, 1 and 2 to
  discourage directly embedding fitted numerical estimates.
- The common numerical snapshot evaluator retains responsibility for finite,
  nonnegative rate predictions over evaluated points and uncertain parameter
  particles. Parsing is explicitly not a global domain or semantic certificate.

The message constructor validates numeric histories, including legitimate
negative noisy log1p-rate observations. It does not silently convert these to
exact nonnegative rates or include uncontrolled narrative instructions.

## Executed verification

80 focused tests passed in 4.28 seconds across structure proposals, actual symbolic
search, executable inference and sealed prediction. Scoped lint passed.

New tests cover blind-prompt invariance, outcome-responsive prompt content,
nonfinite and nonnumeric rejection, strict schema and duplicate JSON keys,
unsafe expressions, unchanged externally supplied priors, canonical deduplication,
new-law history replay with parameter uncertainty, and whole-batch failure.

A complete three-arm integration test uses fixture LLM response strings and real
pinned gplearn symbolic search on a constructed linear-rate history. All three
arms fit that same history, seal predictions for the same targets, and only then
invoke the outcome loader once. The scorer grants no scientific or paid authority.
This test proves integration, NOT that either LLM proposed the fixture or outperforms
the symbolic method. No source-world experiment or closed interface was executed.

## Remaining limitations and decision

This is one missing part of deliverable C, not completion of C or the full plan.
The chemistry opportunity formulation remains closed after its numerical nulls.
Number Game's initial-support null is also unchanged. None of this infrastructure
authorizes calls on those formulations.

A future prospectively frozen runner must bind the same public context/prior to
both message construction and response parsing, record exact raw responses and
costs, enforce the account-wide daily cap before dispatch, keep test outcomes
sealed, and fail complete cases on unusable proposals. The current module cannot
prove a caller used real rather than simulated histories or never inspected an
outcome earlier. It exposes no such outcome input, but protocol isolation is still
required. Its validation does not supply a hard process wall-time interrupt.

Shared priors do not correct data-dependent structure selection. Literal restrictions
do not prove an expression cannot encode a fitted constant algebraically. Priors,
grammar and variable scaling may limit predictive quality. They must be evaluated
prospectively, not tuned on a failed endpoint. Fixed structures within each planning
pass still do not model future LLM structural refresh.

The next scientific dependency remains a justified new source/opportunity route
and useful calibrated proposals versus history-blind and productive symbolic
controls. No fresh model-quality claim, paid permission or automation resumption
is inferred from these green mechanics tests.
