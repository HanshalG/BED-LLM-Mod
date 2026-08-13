# Tau2 Native-Prerequisite Semantic Mechanics Protocol

Date frozen: 2026-08-13

## Claim Boundary

The official Tau2 population audit established a source-level depth-two gap on
44/44 opportunity episodes. This mechanics stage asks a narrower question:
can one frozen LLM semantic forward-model interface represent the observable
tool behavior well enough for exact Bayesian lookahead on the six already
authorized mechanics episodes?

This stage is calibration, not policy efficacy. It opens no development,
confirmation, reserve, repair-success, or task-success endpoint. A pass may
authorize only a separately frozen development protocol. Any failure closes
this exact interface, model, mechanics cohort, and seeds without prompt repair,
resampling, threshold repair, or endpoint access.

## Irreducible Model Role

Code supplies a finite semantic support, shared conditions, legal action graph,
and typed observable fields. The LLM supplies the joint semantic forward model:
for every candidate state and legal action it predicts one typed observable
signature and a confidence. Code converts those predictions into normalized
likelihoods, performs Bayes updates, and computes exact one- and two-step
information values. The official Tau2 environment is used only after all six
raw model responses are durably banked, to score calibration and source rank.

The model receives no selected task ID, source fault identifier, raw selected
tool response, hidden true state, repair action, task reward, entropy, EIG,
preferred action, or endpoint outcome. Human-readable candidate descriptions
are deliberately visible because semantic state-to-observation prediction is
the model's scientific role.

## Typed Observation Codec

Each response contains every supplied `world_id` exactly once and, for every
action, exactly the machine-specified fields plus one confidence in `[0.50,
0.95]`. Fields are booleans or closed enums. They cover APN state, app presence,
MMS success, network mode/status, speed, status-bar state, Wi-Fi Calling,
messaging permissions, customer/account lookup, payment request, SIM state,
bills, data usage, and line details. No free-text outcome label is accepted.

For one action, deduplicate the LLM's typed point signatures across candidate
states and add an `OTHER` category. If there are `K` categories, a candidate
assigns its confidence `c` to its predicted signature and distributes `1-c`
uniformly over the other `K-1` categories. This is the complete normalized
likelihood used for planning and realized Bayes updates. An official typed
signature absent from the predicted set maps to `OTHER`. No clipping,
renormalization, repair, or source-informed relabeling is allowed.

## Frozen Execution

- Model: exact `deepseek/deepseek-v4-flash-0731`.
- Reasoning: disabled; temperature `0.0`.
- Requests: exactly six, one per mechanics episode, dispatched as concurrent
  schema-homogeneous MMS and mobile sub-batches with aggregate concurrency six.
- Seeds: `202608130100` through `202608130105` in episode order.
- Maximum output: 6,000 tokens per request.
- Retry allowance: zero.
- Structured output: strict JSON Schema.
- Concurrency: six.
- Stage cap: `$0.10`; hard account-wide Europe/London cap: `$5.00`.
- Frozen Aug-13 cumulative-usage boundary: `$220.134128880`.
- Before every HTTP attempt, reserve `$0.01`; the complete `$0.10` stage cap
  must fit before the first request. Live catalog and cumulative usage are read
  in preflight and immediately again before ledger creation and dispatch.
- Reconciled spend is the maximum of posted usage since the boundary and local
  accepted-request cost. Unrelated account use counts; unused allowance does
  not roll over.

The producer checkpoints all six raw responses and prompt/privacy hashes before
loading official mechanics worlds or executing any selected tool. A partial
batch, malformed response, provider error, budget race, binding mismatch, or
account inconsistency banks a terminal failure and authorizes nothing.

## Frozen Gates

All gates are conjunctive.

### Integrity and serving

1. Exactly six accepted requests and six HTTP attempts; zero retries,
   reasoning tokens, forced exits, provider errors, or schema repairs.
2. Every response strictly matches the frozen world/action/field order and
   confidence range; all likelihood rows normalize and all scores are finite.
3. Prompt privacy, response-before-source ordering, source/artifact bindings,
   daily accounting, and unopened downstream paths all replay exactly.

### Semantic calibration

4. Across all 234 mechanics world-action cells, typed-signature accuracy is at
   least `0.80` and mean multiclass Brier score is at most `0.20`; each of the
   three families separately has Brier at most `0.25`.
5. Across the 26 native follow-up cells (`messaging_permissions` or
   `line_details`), at least 24 signatures are exact, at least 24 realized
   posteriors rank the true state first (ties count), mean true-state posterior
   is at least `0.60`, and mean posterior Brier is at most `0.20`.
6. For source-equivalent state pairs under an action, mean predicted
   total-variation distance is at most `0.05` and the maximum is at most `0.15`.
   This directly rejects hidden-state leakage into observationally identical
   tools.

### Planning fidelity

7. In all six episodes, semantic greedy chooses a non-prerequisite root and
   semantic depth two chooses the native prerequisite.
8. Every episode has semantic horizon gain at least `0.05` nats and mean gain
   is at least `0.15` nats.
9. Pooled Spearman correlation between semantic and exact source two-step root
   values is at least `0.70` over all 44 episode-root cells.

A pass is `semantic_mechanics_pass` and authorizes only prospective development
freezing. Otherwise the terminal status is `semantic_mechanics_null` or
`failed_closed`; no policy endpoint opens.

## Controls Required Downstream

Any later development must be frozen before its responses and use paired common
random numbers. It must compare adaptive depth two against greedy one-step,
compute-matched myopic ensemble, and random controls; include a separately
labelled thinking/naive baseline; score entropy AUC, truth log posterior, and
terminal Brier; and retain sealed confirmation outcomes. This mechanics pass,
if obtained, cannot itself support an efficacy headline.
