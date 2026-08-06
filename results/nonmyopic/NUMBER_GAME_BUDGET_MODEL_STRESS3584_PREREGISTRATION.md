# Number Game Budget-Model Stress-3584 Preregistration

Date frozen: 2026-08-06, before either reliability-128 model result and before
any stress seed was opened.

## Purpose

Test whether the selected budget model can generate strict, executable,
history-conditioned Number Game supports reliably at policy-scale volume. This
is a mechanics gate, not a policy-efficacy result. It addresses the observed
long-tail failures directly: GPT-5.6 Luna produced a fatal malformed response
after 1,200 scale requests, and DeepSeek V4 Flash 0731 produced one conditioned
support with zero valid hypotheses in an exact-10 screen.

## Preconditions And Selection

The August 7 Qwen history-blind control must be complete, mechanics-passing,
and independently verified. Its ledger must contain the frozen waiting
`number-game-budget-model-stress3584-1` authorization with a `$1.55` cap.

Both disjoint reliability-128 gates must be banked first. A model is eligible
only when its public result has status `passed` and every frozen gate passes.
If neither is eligible, the stress gate makes zero model calls and closes as
`no_eligible_model`. If one is eligible, select it. If both are eligible,
maximize this lexicographic mechanics tuple:

1. conditioned-support minimum;
2. conditioned-support mean;
3. fewer initial parse failures;
4. fewer format-retry requests;
5. fewer forced exits;
6. fewer transport retries;
7. lower measured cost;
8. fixed final tie priority: DeepSeek 0731, then Luna.

No policy endpoint, Brier value, selected root, or scientific result enters
model selection.

## Models And Seeds

- `openai/gpt-5.6-luna`: request seeds `1081001..1081016`;
- `deepseek/deepseek-v4-flash-0731`: request seeds `1081101..1081116`;
- aggregate concurrency `64`, sixteen seed groups with concurrency four;
- temperature and strict 24-item response schema unchanged from reliability128;
- reasoning disabled.

## Frozen Cases

The source is the hash-bound August 6 Qwen source `TREES.json`, SHA-256
`f812e4a356f5129a6f2f22d5f0f995b76b4ac624a0f312154064ff8c51f2c7b0`.
Case-selection seed is `1081200`.
The resulting canonical case manifest SHA-256 is
`017ef5554906954e8514be487de4cb1924bbf19f295d9977953e486891bb71cd`.

Construct exactly 3,584 cases:

- 16 initial-support cases, one per model seed group;
- every 92 one-observation and 598 two-observation history not used by the
  frozen reliability128 gate, exactly once;
- 2,878 deterministic repeat draws distributed across the 690 unseen
  conditioned histories under different seed groups.

Unique unseen histories are ordered by SHA-256 of the selection seed and
encoded history. Unique history `i` uses group `i mod 16`. Repeat draws cycle
through the ordered histories; each draw takes the first cyclic group with
remaining quota that has not already served that history. Thus every group
receives exactly 224 cases and no history reuses a model seed. The enlarged
sample has about 95% probability of observing a failure whose true rate is one
per 1,200 requests, versus about 57% for 1,024 requests.

## Retry And Mechanics Gates

Strict top-level parse failures are retried once with the same prompt,
temperature, and group seed only when at most eight initial responses fail.
Semantic filtering failures are not retried.

All gates are required:

- exactly 3,584 final case records;
- at most eight initial strict-parse failures, all and only those retried once;
- every final response strictly parses and contains exactly 24 schema items;
- accepted requests equal `3584 + format retries`;
- HTTP attempts equal accepted requests plus transport retries;
- at most 16 transport retries and zero provider-error retries;
- zero reasoning tokens and at most eight forced exits;
- all 16 initial supports contain at least 16 valid unique hypotheses;
- all 3,568 conditioned supports contain at least four valid unique
  hypotheses, with mean at least eight;
- locally measured run cost at most `$1.55`.

Repeated-history support Jaccard similarity is descriptive only. Diversity is
not treated as failure because prior evidence found independent support-draw
diversity useful for planning.

## Budget And Artifacts

Before adapter construction, require that both reliability tail entries are
no longer pending, the stress authorization is exact, live balance is at least
`$1.55`, and the shared Europe/London ledger has at least `$1.55` remaining.
The shared OpenRouter run ID enforces the aggregate `$1.55` cap across all
sixteen adapters. Locally measured and posted account spend are reconciled on
success and failure.

Publish the model selection inputs, frozen protocol, mechanics gates,
conditioned-support summary, repeated-history descriptives, per-case extension
hashes, usage, and hashes of both reliability results and private raw
responses. Keep raw response text private and untracked.
