# UCI Thyroid Utility-Grounded GPT-5.4 Mini Preregistration

Status: frozen after the names-only GPT-5.4 Mini trajectory result and before any
utility-grounded GPT-5.4 Mini response or scientific endpoint is observed.

## Question and treatment

The names-only model completed all trajectories and beat exact depth one, but it did
not beat matched-random continuations. At the identical initial belief it proposed
`query:age` after blood collection in 48/50 trajectories and `query:tsh` only twice.
This experiment tests the localized mechanism: whether calibrated continuation
utility, rather than a larger model or a longer horizon, repairs the proposal prior.

The only treatment is an opt-in branch-local utility card. For every fixed root
outcome and every legal continuation, the prompt includes:

- posterior-predictive expected class entropy after the continuation; and
- the equivalent one-step information gain from that branch posterior.

These values are computed from the 7,200-row empirical prior and the deployed
history. They use neither the held-out patient's row nor a realized future outcome.
The LLM still returns one named continuation per branch. The exact depth-two
verifier still scores each complete strategy and chooses among the machine-fixed
collection plus top-three immediate-query roots. This is a utility-grounding
mechanism ablation, not evidence that the LLM inferred assay utility unaided.

## S0 late-state serving smoke

- Seed `24159`; the same 12 history lengths `0,1,2,3,4,5,6,6,5,4,3,2` used by the
  prior robust smoke.
- Model `openai/gpt-5.4-mini` through OpenRouter, non-thinking, temperature zero,
  1,024-token output cap, one bounded validation-feedback retry.
- Required: all 12 accepted; lengths 0--6 covered; every named continuation complete
  and legal; no continuation repeats its root; utility cards logged; zero reasoning,
  forced exits, and scoring-time LLM calls.
- S0 is mechanics-only. Any failure stops this line without parser or prompt repair.

## Conditional S1 paired confirmation

S1 runs only if S0 passes every mechanic.

- Fresh seed `24160`; 50 patient rows without replacement; eight paired actions.
- Arms: utility-grounded GPT named depth-two continuations, matched-random named
  continuations on identical roots, exact depth one, and exhaustive depth two. All
  arms use the same exact one-step rule on the final action.
- Primary endpoint: mean post-action target-entropy AUC. Corroborating endpoint:
  truth-log-posterior AUC. Ten thousand paired bootstrap replicates.
- Exactly 350 accepted logical model cells; utility summaries and physical requests
  retained for audit.

All frozen S1 gates are required:

1. Positive paired 95% lower bounds for entropy-AUC and truth-log-AUC gains over
   exact depth one.
2. Positive paired 95% lower bounds for both gains over matched random.
3. At least 60% recovery of exhaustive depth two's mean entropy gain over depth one.
4. Blood collection selected first on at least 75% of trajectories.
5. Fifty distinct paired truths, complete legal traces, exactly 350 accepted cells,
   and zero reasoning, forced exits, and scoring-time LLM calls.
6. Independent full replay validates every history, observation, posterior, policy
   score, selected action, control, aggregate, and fresh-bootstrap gate.

Failure after the one registered retry stops without replacement or alternate seed.
The OpenRouter run cap remains `$6`; based on the names-only run, S0 plus S1 should
cost below `$0.60`. Project spend is `$34.92378331` of `$110`, leaving
`$75.07621669` before this experiment.
