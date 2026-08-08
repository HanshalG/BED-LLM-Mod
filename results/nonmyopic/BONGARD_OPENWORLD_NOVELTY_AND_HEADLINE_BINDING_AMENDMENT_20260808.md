# Bongard OpenWorld Novelty and Headline-Binding Amendment

Frozen: 2026-08-08, before any Bongard serving, mechanics, development, or
confirmation response and before any scientific endpoint was opened.

This amendment fixes two reporting issues found in a zero-call primary-source
audit. It changes no model, prompt, response schema, task, partition, seed,
action, endpoint, policy, score, gate, request count, or budget.

## Closest-Work Boundary

The paper does not claim that it is the first work to combine an LLM with
Bayesian experimental design, LLM-generated hypotheses, or formal non-myopic
planning. The closest distinctions are narrower:

- BED-LLM regenerates and filters hypotheses at observed turns, but optimizes
  incremental one-step EIG and describes final-belief planning as generally
  computationally infeasible.
- *Wild Guesses and Mild Guesses* repeatedly proposes executable program
  particles with an LLM. Its one-step EIG is computed on the current particle
  posterior and explicitly does not account for generator failures after the
  update; the paper identifies that omission as a proposal-support trap.
- Zero-Shot Active Feature Acquisition via LLM-Elicitation performs formal
  non-myopic acquisition using an LLM-elicited MRF, but planning conditions a
  fixed elicited model rather than valuing answer-conditioned model
  regeneration.
- ASIG trains a multi-turn policy but calculates its EIG reward only for the
  first question in the reported setup.

Relative to these closest works, the Bongard claim is only that the frozen
experiment explicitly plans over the LLM's answer-conditioned future
predictive-belief matrices and tests that value against one-step, fixed-support,
same-seed history-blind, matched fixed-score, matched realized-updater,
shuffled, and random controls. This claim is made only if the full untouched
confirmation conjunction passes. It is not a universal priority claim over
every active-learning or learned-likelihood method.

Primary sources:

- BED-LLM: https://openreview.net/forum?id=qyylZMLYT8
- Wild Guesses and Mild Guesses: https://arxiv.org/abs/2602.06818
- Zero-Shot Active Feature Acquisition: https://arxiv.org/abs/2606.18933
- ASIG: https://arxiv.org/abs/2607.03426

## Deterministic Headline Binding

The prior paper-fragment renderer could authorize a Bongard headline while
placing the result only in a late body paragraph. The renderer must now emit a
second zero-call TeX artifact defining two macros:

- `\BongardAbstractResult`
- `\BongardContributionResult`

For development, confirmation null, or confirmation mechanics failure, both
macros are empty. Only `full_llm_native_confirmation` may populate them. Its
abstract sentence must state the untouched task count, complete registered
conjunction, primary dynamic-versus-one-step relative Brier reduction and
paired interval, and superiority over the registered fixed-support,
history-blind, and matched-updater families. Its contribution item states the
same qualitative scope without adding an unregistered endpoint.

The main manuscript contains inert hooks for these macros and a separate body
fragment hook. The renderer records hashes for both generated TeX artifacts.
Unknown tiers, stale manuscript/reference hashes, missing replay, non-finite
metrics, or an unauthorized nonempty headline fail closed. No hand-written
result-contingent abstract or contribution edit is permitted.
