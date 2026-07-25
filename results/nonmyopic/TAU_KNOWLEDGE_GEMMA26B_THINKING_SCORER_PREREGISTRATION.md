# tau-Knowledge Gemma 4 26B Thinking Scorer Replication

## Status

Frozen before any Gemma scorer response. This is a post-hoc cross-model
replication on the already open V3.1 trees, not a fresh-task policy test.

## Question

Can Gemma 4 26B A4B with thinking reproduce the semantic root and
continuation-ranking mechanism that GPT-5.4 exhibited on tau-Knowledge?

## Frozen Interface

- Model: `google/gemma-4-26b-a4b-it` through OpenRouter.
- Thinking enabled with 4,096 first-pass tokens.
- One reasoning-disabled 1,024-token forced-final request is allowed only by
  the existing repaired adapter after an empty/provider-notice length stop.
- Temperature zero, 256 concurrency ceiling, no response repair or score
  coercion beyond the already frozen permissive numeric parser.
- Same two smoke trees and 20 confirmation trees, prompts, path-dependent
  beliefs, queries, retrieved documents, endpoints, and tie breaking as the
  GPT-5.4 V3.1 scorer.
- No tree, query, retrieval, belief, or endpoint regeneration.

The source smoke artifact SHA-256 is
`61c466ead49a5545abd54dd28e1a2fd7ad070287f5108684a3b2643befc40a01`;
the confirmation artifact SHA-256 is
`f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae`.

## Stages

### Serving and efficacy smoke

Exactly 14 logical prompts:

- two myopic root scorers;
- two full-tree non-myopic root scorers; and
- ten focused continuation scorers.

The smoke passes only if:

- all 14 logical responses parse without repair;
- physical requests are between 14 and 28;
- reasoning tokens are positive;
- every forced exit has exactly one successful forced-final request;
- scores vary on at least 8/10 focused roots;
- focused pairwise accuracy is at least `.55`;
- at least 7/10 focused choices are oracle-optimal; and
- cost is at most `$0.25`.

### Conditional confirmation

Only a passing smoke authorizes exactly 140 logical prompts on the 20 frozen
confirmation trees. Physical requests may not exceed 280, every forced exit
must finalize successfully, and cost may not exceed `$2.00`.

All original V3.1 confirmation efficacy gates remain:

- non-myopic root accuracy at least `.60` and gain over the freshly scored
  myopic root at least `.05`;
- focused accuracy at least `.60`, optimal rate at least `.70`, mean regret at
  most `.30`, and selected-root continuation loss at most 5 documents;
- at least four wins, at most two losses, and total gain at least four
  documents over myopic;
- at least six wins, at most four losses, and total gain at least five over
  random; and
- at least four documents gained over joint continuation selection.

It must also beat the strongest frozen nonsemantic controls: root accuracy
above `.5455`, continuation accuracy above `.6316`, and endpoint total at
least `25`.

## Interpretation

Passing supports cross-model transfer of the semantic scorer under a
thinking-enabled, compute-unmatched model. It does not establish fresh-task
generalization, belief-alignment causality, or a fair compute comparison to
GPT-5.4. Failure closes this exact Gemma interface without prompt, budget,
parser, threshold, or model-size repair.

## Budget

Smoke projected below `$0.10` with a `$0.25` cap; conditional confirmation
projected below `$0.75` with a `$2.00` cap. Launch requires preserving the
protected `$25` live reserve. OatML remains paused.
