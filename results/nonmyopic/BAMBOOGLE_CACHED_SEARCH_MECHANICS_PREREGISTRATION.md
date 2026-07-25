# Bamboogle Cached-Search Semantic-Belief Mechanics Preregistration

## Purpose

Test whether a non-reasoning LLM can maintain an evidence-sensitive semantic
answer belief and use a second, observation-conditioned Wikipedia query to
make a useful non-myopic first-query choice. This is an interface/mechanics
gate on the five already released Bamboogle records. It is not a paper result
and cannot release the opportunity, development, or holdout splits.

## Frozen Data And Model

- Tasks, in manifest order:
  `test_87`, `test_110`, `test_72`, `test_69`, `test_61`.
- Source file SHA-256:
  `c9703dae6bb1ceb9e2df77be45da28cb12aa040d2f471507890a461296968f3f`.
- Model: `openai/gpt-5.4` through OpenRouter.
- Reasoning: disabled.
- Temperature: `0.0`.
- Scientific/model retries and response repairs: zero.
- Maximum OpenRouter cost: `$0.75`.
- OatML use: none.

Gold answers are absent from all model and Wikipedia requests. They are loaded
only after every model response and retrieval result has been checkpointed.

## Frozen Semantic State

Every belief state contains exactly eight distinct short answer hypotheses and
eight integer weights. Each weight is in `1..100` and the weights sum to
exactly `100`. Entropy is the categorical entropy of these weights in nats.
The support is open-world: the model regenerates all eight hypotheses after
each observation rather than aligning probabilities onto a frozen list.

For each task, one initial call emits:

- the initial eight-hypothesis belief;
- four distinct root Wikipedia search queries; and
- one distinct precommitted second query paired with each root.

The eight queries emitted in this call must all be distinct after
normalization.

## Frozen Retrieval And Updates

Each query uses the English Wikipedia MediaWiki API with `generator=search`,
top `3` pages, introductory plaintext extracts, and a fixed user agent.
Extracts are whitespace-normalized and truncated to `1,200` characters per
page before entering prompts. Every response is cached verbatim with its
parsed representation and SHA-256. Transport-only retrieval retries are
allowed up to two times; they do not change a query.

For each of the `20` root branches:

1. execute the root query;
2. call the model once to regenerate the eight-hypothesis belief and emit one
   adaptive second query after seeing the root results;
3. execute both the adaptive second query and the paired precommitted second
   query;
4. call the model once to regenerate the terminal belief from root plus
   adaptive evidence; and
5. call the model once to regenerate the terminal belief from root plus
   precommitted evidence.

This gives exactly `65` logical and physical model calls:
`5` initial, `20` root refresh, `20` adaptive terminal, and `20`
precommitted terminal. There are exactly `60` logical retrieval actions.
Repeated identical queries may use the frozen cache, so physical MediaWiki
request count can be lower.

## Frozen Policies

For a belief with probabilities `p`, define
`H(p) = -sum_i p_i log(p_i)`. Ties always select the lowest root index.

- `myopic`: root with maximum initial-to-root entropy reduction.
- `adaptive_d2`: root with maximum initial-to-adaptive-terminal entropy
  reduction.
- `fixed_d2`: root with maximum initial-to-precommitted-terminal entropy
  reduction.
- `random`: one root per task from Python `random.Random(24406)`.

The `myopic`, `adaptive_d2`, and `random` policies are evaluated on the
adaptive terminal belief of their selected root. `fixed_d2` is evaluated on
the precommitted terminal belief of its selected root. Thus the primary
comparison changes first-query selection, while myopic and adaptive depth-2
share the same observation-conditioned second-query execution.

## Frozen Metrics

Target-blind:

- initial, root, adaptive-terminal, and precommitted-terminal entropy;
- selected root under each policy;
- count of adaptive queries that differ from paired precommitted queries;
- count whose top returned page differs;
- count of root beliefs that differ from the initial belief; and
- per-task terminal entropy range across adaptive roots.

Endpoint-only, computed after checkpointing:

- exact normalized gold-answer mass in every terminal belief;
- exact correctness of the maximum-weight hypothesis;
- policy-level gold mass and top-answer correctness; and
- per-task gold-mass range across adaptive roots.

Normalization lowercases, removes ASCII punctuation and articles
`a`/`an`/`the`, and collapses whitespace. Public output omits questions,
answers, hypotheses, queries, extracts, and raw responses.

## Conjunctive Mechanics Gate

All conditions must hold:

1. exactly `65` physical model requests and `65` HTTP model attempts;
2. zero model retries, reasoning tokens, forced exits, repairs, or parse
   failures;
3. exactly `60` logical retrieval actions, all with nonempty parsed results;
4. all `20` root beliefs differ from their task's initial belief;
5. at least `15/20` adaptive queries differ from their paired precommitted
   queries;
6. at least `10/20` adaptive and precommitted queries return different top
   pages;
7. adaptive terminal gold-mass range is at least `.25` on at least `3/5`
   tasks;
8. `adaptive_d2` selects a different root from `myopic` on at least `2/5`
   tasks;
9. `adaptive_d2` gold mass strictly exceeds `myopic` on at least one task and
   is lower on at most one task;
10. `adaptive_d2` gold mass strictly exceeds `fixed_d2` on at least one task
    and is lower on at most one task; and
11. total OpenRouter cost is at most `$0.75`.

Failure closes this exact interface without threshold repair. Passing
authorizes only a separately committed 20-task opportunity protocol; it does
not authorize development or holdout access.
