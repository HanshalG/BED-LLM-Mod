# tau-Knowledge Gemma 4 26B Compact Thinking Replication

## Status

Frozen before any response under this interface. This is a post-hoc
cross-model replication on the already open V3.1 trees, not a fresh-task policy
test.

## Motivation

The first Gemma 4 26B thinking smoke completed 13/14 strict objects and showed
focused pairwise accuracy `.6538` on the nine valid focused roots. It failed
serving because one 4,096-token completion ended after a nonempty partial,
rationale-shaped JSON prefix. That result did not measure the tenth root or
authorize confirmation.

This V2 changes the output contract rather than the semantic task. The same
inputs now request only compact arrays of scores and selected followup indices,
with no rationales. The larger first-pass budget tests whether Gemma transfers
the semantic ranking mechanism when serialization length is not the
bottleneck.

## Frozen Interface

- Model: `google/gemma-4-26b-a4b-it` through OpenRouter.
- Thinking enabled with 8,192 first-pass tokens.
- One reasoning-disabled 512-token forced-final request is allowed only after
  an empty or recognized provider-notice length stop.
- Temperature zero and concurrency ceiling 256.
- Exact same two smoke trees, 20 confirmation trees, prompts' semantic
  instructions, path-dependent beliefs, queries, retrieved documents,
  endpoint labels, policy controls, and tie breaking as V3.1.
- No tree, retrieval, belief, score, or endpoint from the failed V1 Gemma
  responses is reused.
- Root output is exactly `{"scores":[...five...]}`
  and, for full trees, `{"scores":[...],"best_followups":[...five...]}`.
- Focused output is exactly `{"scores":[...four...]}`.
- Integers and at-most-two/three-digit strings use the already frozen
  cross-model numeric coercion; no extra keys, prose, missing values, or
  out-of-band focused scores are accepted.

Source hashes:

- smoke tree:
  `61c466ead49a5545abd54dd28e1a2fd7ad070287f5108684a3b2643befc40a01`;
- confirmation tree:
  `f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae`;
- frozen nonsemantic controls:
  `d1d43ecf0f09fd5874e9badd6cff4e9ae5c3da882a3ea408b816225c2d2b29f4`.

## Serving And Efficacy Smoke

Exactly 14 logical prompts:

- two myopic root scorers;
- two full-tree non-myopic root scorers; and
- ten focused continuation scorers.

The smoke passes only if:

1. all 14 logical responses parse without repair;
2. physical requests are between 14 and 28;
3. reported reasoning tokens are positive;
4. every forced exit has exactly one successful forced-final request;
5. at least 8/10 focused score vectors vary;
6. focused pairwise accuracy is at least `.55`;
7. at least 7/10 focused choices are oracle-optimal; and
8. total cost is at most `$0.25`.

Failure closes compact Gemma V2. No output-budget, prompt, parser, score-band,
threshold, or task repair follows.

## Conditional Confirmation

Only a passing smoke authorizes exactly 140 logical prompts on the same 20
open confirmation trees. Physical requests may not exceed 280 and cost may not
exceed `$2.00`.

All original V3.1 efficacy gates remain:

- non-myopic root accuracy at least `.60` and gain over the freshly scored
  myopic root at least `.05`;
- focused accuracy at least `.60`, optimal rate at least `.70`, mean regret at
  most `.30`, and selected-root continuation loss at most five documents;
- at least four wins, at most two losses, and total gain at least four
  documents over myopic;
- at least six wins, at most four losses, and total gain at least five over
  random;
- at least four documents gained over joint continuation selection; and
- root accuracy above `.5455`, continuation accuracy above `.6316`, and
  endpoint total at least 25 against the strongest frozen nonsemantic controls.

Passing supports cross-model transfer of semantic root and continuation
ranking under a thinking-enabled, compute-unmatched model. It does not establish
fresh-task generalization, correct-belief causality, or a fair compute
comparison to GPT-5.4.

## Budget

The smoke is projected below `$0.10` with a hard `$0.25` cap. Confirmation is
projected below `$0.75` with a hard `$2.00` cap. Live credits and the protected
reserve must be checked before each stage. OatML remains paused.
