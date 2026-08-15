# Number Game DeepSeek Stratified Semantic V4 Protocol

Date frozen: 2026-08-15 (Europe/London), after terminal closure of Qwen V3 and
before any DeepSeek V4 Number Game response, run artifact, or paid request.

## Purpose

Test the first link that closed V3: can a stronger inexpensive text model
generate open-ended executable Number Game hypotheses while obeying mutually
exclusive five-anchor signature strata? This is a semantic transport gate, not
a planning or efficacy experiment. A pass authorizes only a separately frozen
fresh mechanics protocol; a failure closes this model/interface.

The LLM remains irreducible. It writes every rule expression and name. Code may
parse, execute, validate history and signature membership, reject, and measure
diversity. Code may not generate, repair, mutate, patch, translate, complete, or
resample a response.

## Model And Calls

- exact model: `deepseek/deepseek-v4-flash-0731`;
- reasoning disabled explicitly; no reasoning-token fallback;
- temperature `0.6`;
- strict JSON object with exactly `name` and `expression`;
- maximum output `220` tokens;
- zero retries and no forced finalization;
- concurrency at most `64`;
- fresh seeds `202608210000..202608210191` in canonical group/slot order;
- exactly `192` accepted requests: 64 per public history group.

Each group has two requests per signature. Slots 0--1 request `00000`, slots
2--3 request `00001`, and so on through slots 62--63 requesting `11111`.
Requests are independent and checkpointed before any parsing decision can
authorize another group.

The current authenticated endpoint catalog has active exact-model providers
supporting seed, response format, and explicit reasoning disable at or below
`$0.14/M` prompt and `$0.28/M` completion tokens. A price increase can only stop
the run. It cannot raise the price ceiling or budget.

## Public Semantic Groups

Use the frozen V3 DSL, parser, anchor order, signature order, system prompt, and
response schema exactly. The three groups are:

1. `initial`: no observations;
2. `one_step`: observation `2 = YES`;
3. `two_step`: observations `2 = YES`, `3 = NO`.

The protected query numbers are the history query numbers. The resulting five
anchors are `(7,12,25,42,81)` in every group. An accepted rule must execute on
all integers `0..100`, obey every observation, and match all five requested
anchor memberships exactly.

Before this freeze, the already frozen 416,366-extension grammar bank was used
only as a zero-call source feasibility audit. All 32 strata are nonempty in all
three groups. Minimum per-stratum counts are respectively `2188`, `402`, and
`120`; the grammar-bank SHA-256 remains
`6e2a523d7d7c0c59b5df0494de37087525c7009d43a5eaba01390e005d85e0e0`.
This audit influenced the public histories, so grammar novelty has no gate
authority and the grammar bank never enters a prompt or response weight.

## Frozen Gates

All conditions are conjunctive.

### Transport

1. Exactly 192 accepted requests and HTTP attempts, exact model and seed set,
   zero retries, reasoning tokens, forced exits, or provider errors, and every
   response has clean `stop` finish.
2. Every prompt, history, protected set, anchors, signature, schema, and payload
   hash replays independently from public constants.
3. The complete stage costs at most `$0.20` and respects the account-wide Aug 15
   `$5.00` cap from opening usage `$220.339269126`; unrelated usage and the
   posted V3 cost `$0.007297280` count.

### Semantic obedience and diversity

4. Every group retains at least 48/64 executable, history-consistent,
   signature-consistent particles.
5. Every group covers at least 28/32 distinct requested signatures and at least
   24 unique executable extensions.
6. In each group, each parity half (one request per signature) retains at least
   24/32 valid particles, covers at least 24 signatures, and contains at least
   20 unique extensions.
7. Across all accepted particles, maximum within-group extension multiplicity
   is at most four. Different signatures must always imply different executable
   extensions by exact semantics.

Names, expression-family frequencies, grammar novelty, and cross-group overlap
are descriptive only. No canonical target, candidate root, simulated answer,
policy score, Brier endpoint, development row, or confirmation row may be
loaded by this gate.

## Budget And Failure

Use a fresh Aug 15 run, binding, ledger component, and terminal path. Before
each HTTP block, reread authenticated credits/usage and the exact endpoint
catalog, reserve the complete byte-as-token worst-case block exposure, and
require it to fit the `$0.20` stage cap, remaining daily allowance, and account
balance. Reauthorize immediately before every HTTP attempt. Reconcile the
larger of posted usage since the frozen opening and local accepted-request cost.

Any malformed response, transport mismatch, floor failure, budget loss, replay
failure, or gate null is banked once and authorizes nothing. There is no retry,
seed substitution, prompt repair, threshold change, favorable-subset scoring,
or response reuse. Synthetic/adversarial tests and a pushed immutable binding
are required before authenticated preflight.
