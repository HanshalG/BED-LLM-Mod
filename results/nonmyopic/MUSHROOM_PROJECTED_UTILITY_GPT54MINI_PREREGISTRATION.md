# UCI Mushroom Projected-Utility GPT-5.4 Mini Preregistration

Status: frozen after implementation, deterministic testing, and replay of the old
names-only result, before any GPT-5.4 Mini response under this Mushroom treatment.

## Motivation and treatment

The independently qualified Mushroom task has a strict zero-information acquisition
opportunity: specimen collection unlocks detailed features. Exact depth two beats
depth one by `+0.1153` entropy-AUC nats with paired 95% CI
`[+0.1057,+0.1254]`. The prior names-only Gemma 4 26B proposal gate beat matched
random overall but selected collection on only `10/16` strict opportunities and
recovered `32.7%` of exhaustive depth two. Its collection continuation was bruises
on nine cells, odor on six, and spore-print color on one.

This fresh treatment leaves the task, four machine-fixed roots, branch-local integer
menus, exact depth-two verifier, matched controls, and one correction response
unchanged. It adds:

1. Every legal continuation under every positive-probability root outcome is
   annotated with empirical posterior-predictive expected class entropy and one-step
   information gain. The cards use only the processed-cohort model and current
   history, never the held-out row or a future observation.
2. After both bounded responses are invalid, valid branch indexes are preserved and
   only invalid or missing branches are projected to the exact minimum-entropy legal
   continuation, using existing `EPSILON` semantics and legal-order tie-break.
3. Prompts, utility cards, raw responses, compiled policies, invalid responses, and
   projection events are retained for independent replay.

This is a machine-grounded LLM-Modulo treatment, not unaided LLM planning.

## S0 serving smoke

- Fresh seed `24173`; ten distinct posterior cells spanning uncollected and collected
  states with histories of length zero through four.
- `openai/gpt-5.4-mini` through OpenRouter, non-thinking, temperature zero,
  128-token output cap, one correction response.
- Required: all ten cells complete and legal; collection represented in every
  uncollected cell; exactly ten logical calls; **zero projected cells**; zero
  reasoning tokens, forced exits, and scoring-time LLM calls.
- Any S0 failure or projection stops this line.

## Conditional S1 proposal-quality gate

S1 runs only if all S0 mechanics pass.

- Fresh seed `24174`; 32 balanced distinct posterior cells: 16 uncollected strict
  exact-depth-two collection opportunities and 16 collected semantic states.
- Exact controls and 5,000 paired bootstraps are unchanged from the old gate.

All requirements must pass:

1. Positive paired 95% lower bound for matched-random-minus-GPT policy cost.
2. Positive paired 95% lower bound for exact-depth-one-root-minus-GPT policy cost on
   uncollected opportunities.
3. Collection-root selection on at least 75% of uncollected opportunities.
4. Mean recovery of the exhaustive depth-two opportunity at least 60%.
5. At most 5% projected logical cells and at most 1% projected branch choices.
6. All 32 cells accepted, balanced, distinct, complete, and exactly scored; zero
   reasoning, forced exits, and scoring-time LLM calls.
7. Independent replay reproduces every compiled policy, projection, exact cost,
   control, aggregate, and gate.

No alternate seed, prompt modification, threshold change, or replacement gate
follows a failure. A pass authorizes only a separately preregistered fresh paired
trajectory confirmation. Expected S0 plus S1 cost is below `$0.60`, within the
existing `$6` run cap. Registered project spend is `$37.22902951 / $110`, leaving
`$72.77097049`.
