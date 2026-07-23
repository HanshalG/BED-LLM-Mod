# Cleveland Heart Workup Projected-Utility GPT-5.4 Mini Preregistration

Status: frozen after implementation and deterministic testing, before any response
from GPT-5.4 Mini under this Heart utility-grounded interface.

## Motivation and treatment

The processed 297-row Cleveland task has an independently audited exact depth-two
gain over depth one: entropy-AUC `+0.068747` with paired 95% CI
`[+0.057311,+0.080303]`, truth-log-AUC `+0.068747`
`[+0.047312,+0.090220]`, and workup ordered `2.222` rounds earlier. The prior
names-only Gemma 4 26B proposal gate failed: it selected the zero-information workup
root on only `1/16` exact depth-two opportunities and recovered `-156%` of the
available gain.

This fresh treatment leaves the task, four machine-fixed roots, local integer action
menus, exact depth-two policy verifier, and one bounded correction response
unchanged. It adds:

1. For every positive-probability root outcome, each legal continuation is annotated
   with empirical posterior-predictive expected class entropy and one-step
   information gain. These cards use only the current history and processed-cohort
   empirical model; they contain no held-out truth or future observation.
2. If both model responses are invalid, valid branch indexes are retained and only
   invalid or missing branches are projected to the exact legal minimum-entropy
   continuation, with existing `EPSILON` semantics and legal-order tie-break.
3. Every prompt card, accepted response, invalid response, compiled policy, and
   projection event is retained for independent replay.

This is a machine-grounded LLM-Modulo treatment, not an unaided language-model
planner. Projection is bounded by contribution gates so it cannot silently author
the scientific result.

## S0 serving smoke

- Fresh seed `24165`; ten deterministic Heart posterior cells, five before and five
  after workup, with history lengths zero through two.
- Model `openai/gpt-5.4-mini` through OpenRouter, non-thinking, temperature zero,
  128-token output cap, one correction response.
- Required: all ten cells complete; all roots and follow-ups legal; workup root and
  workup-as-continuation representable; exactly ten logical calls; **zero projected
  cells**; zero reasoning tokens, forced exits, and scoring-time LLM calls.
- Any S0 failure or projection stops this line.

## Conditional S1 proposal-quality gate

S1 runs only if every S0 mechanic passes.

- Fresh seed `24166`; 32 balanced, distinct posterior cells: 16 unworked states where
  exhaustive depth two strictly prefers workup and 16 worked states.
- GPT proposes one complete continuation for each of four fixed roots. The exact
  verifier selects the lowest-cost root policy.
- Controls: matched-random continuations on identical roots, exact depth-one root
  with exact continuation, and exhaustive exact depth two.
- Five thousand paired bootstrap replicates.

All gates are required:

1. Positive paired 95% lower bound for matched-random-minus-GPT policy cost.
2. Positive paired 95% lower bound for exact-depth-one-root-minus-GPT policy cost on
   the 16 unworked opportunities.
3. Workup root selected on at least 75% of unworked opportunities.
4. Mean recovery of the exhaustive depth-two opportunity at least 60%.
5. At most 5% projected logical cells and at most 1% projected branch choices.
6. All 32 cells accepted, balanced and distinct; all policies exactly scored; zero
   reasoning, forced exits, and scoring-time LLM calls.
7. Independent replay reproduces proposals, projections, exact policy costs,
   controls, comparisons, and gates.

No alternate seed, prompt change, threshold change, or replacement gate follows a
failure. The expected S0 plus S1 cost is below `$0.50`, within the existing `$6`
per-run cap. Registered project spend is `$36.64346041 / $110`, leaving
`$73.35653959` before S0.
