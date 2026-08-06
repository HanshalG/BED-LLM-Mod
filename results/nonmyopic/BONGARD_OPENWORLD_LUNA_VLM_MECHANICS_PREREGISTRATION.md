# Bongard-OpenWorld Luna VLM Mechanics Preregistration

Date frozen: 2026-08-06
Earliest execution date: 2026-08-10 (Europe/London)

## Objective

Test whether a cheap multimodal model can serve as the irreducible semantic
belief machinery for sequential BED on the four frozen Bongard-OpenWorld
mechanics tasks. This is a development/mechanics result, not a scientific
confirmation.

DeepSeek V4 Flash 0731 is excluded because the live OpenRouter catalog binds it
as text-only. The first model is `openai/gpt-5.6-luna`, nonreasoning, because it
accepts images and currently costs $0.10/M input and $0.60/M output tokens.
`qwen/qwen3.7-plus` is the frozen escalation model only if Luna fails the
semantic mechanics gate.

## Semantic Belief Contract

For one task/history, Luna receives all 14 images under opaque IDs, four
initial labels, and any simulated or realized query labels. It never receives
source UIDs, filenames, source positions, ground-truth concepts/captions, or
unrevealed labels.

The response contains exactly ten distinct natural-language hypotheses. Each
hypothesis supplies:

- an ID `H01`--`H10`;
- a concise free-form visual rule;
- a positive integer prior weight;
- 14 integer positive-label likelihoods in the supplied opaque image order.

The evaluator normalizes prior weights and applies an analytical Bernoulli
likelihood over only the revealed labels. Thus the VLM irreducibly supplies
both the open-ended support and image likelihoods, while posterior arithmetic
remains exact and independently replayable.

## Exact-10 Serving Gate

Use the first two frozen mechanics tasks by opaque task ID:

- two root histories;
- for the first two opaque candidate IDs in each task, one branch prompt for
  each possible next label;
- total: 2 root + 8 branch = exactly 10 requests.

No actual candidate or endpoint label is accessed. Both branch outcomes are
simulated symmetrically. Responses and image payloads are cacheable but may
not be reused as scientific confirmation data.

The serving gate is conjunctive:

1. exactly 10 accepted requests and 10 HTTP attempts;
2. zero transport/provider retries, reasoning tokens, and forced exits;
3. all 10 strict schemas parse;
4. every response has exactly 10 unique nonempty rules, ordered IDs, positive
   prior weights, and exactly 14 likelihoods in `[1,99]`;
5. every posterior assigns finite mass and finite entropy;
6. aggregate posterior predictions fit the revealed history better than a
   constant 0.5 predictor;
7. positive-versus-negative branch prompts for every candidate produce a
   material change in either rule support or posterior predictions;
8. each root has nonzero candidate EIG and nonidentical candidate scores;
9. no hidden source field or label-bearing path enters any request or public
   result;
10. request cost is at most $0.25 under the current account-wide $5/day ledger.

## Full Four-Task Mechanics Tree

Run only after the exact-10 gate passes. Cache one shared tree per task:

- one root support;
- 16 first-query branch supports (eight candidates times two outcomes);
- one final refreshed support for each distinct selected policy history.

Offline policy controls reuse this exact cache:

- **myopic / compute-matched width:** maximize root one-step EIG, then maximize
  EIG under the realized refreshed branch;
- **fixed-support depth two:** maximize cumulative two-step EIG under the root
  support only;
- **path-dependent depth two:** root EIG plus expected best next-step EIG under
  each branch's regenerated support;
- **shuffled-branch control:** permute regenerated branches across first
  actions before applying the same path-dependent score;
- **random strategy:** two seeded candidates without replacement.

All methods use exact released candidate labels as the environment response and
an independently regenerated final support on their realized six-label
history. Primary development endpoints are mean Brier score and log score on
the untouched official endpoint pair. Accuracy, final entropy, truth-side
probability, support/rule diversity, action changes, and predicted-versus-
realized entropy drop are diagnostics.

The path-dependent score is frozen as:

`root EIG(first) + E_label[max remaining EIG(regenerated branch, second)]`.

Fixed-support depth two uses the same expression with an analytically updated
root support. The myopic width control receives the same cached branch calls
but ignores them when choosing the first action.

## Budget Boundary

The exact-10 gate may run no earlier than Aug 10 and has a $0.25 run cap. A
full mechanics tree requires a fresh preflight based on observed serving cost,
must fit the remaining account-wide $5 allowance, and initially has a $1.50
cap. Failure closes the run before any development or confirmation task opens.
