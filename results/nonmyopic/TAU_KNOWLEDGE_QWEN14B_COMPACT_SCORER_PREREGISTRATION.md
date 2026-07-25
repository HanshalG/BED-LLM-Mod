# tau-Knowledge Qwen3 14B Compact Thinking Replication

## Status

Frozen before any Qwen response. This is a post-hoc cross-family replication
on the already open V3.1 trees, not a fresh-task policy test and not a repair of
the closed Gemma interfaces.

## Question

Can a roughly 15B thinking model reproduce GPT-5.4's semantic ranking of
generated tau-Knowledge retrieval trees?

OpenRouter currently exposes `qwen/qwen3-14b`, not a 15B Gemma endpoint. The
model supports reasoning and structured outputs with a provider completion
ceiling of 8,192 tokens.

## Frozen Interface

- Model: `qwen/qwen3-14b` through OpenRouter.
- Thinking enabled with 7,680 reasoning tokens plus a 512-token answer
  allowance.
- One reasoning-disabled 512-token forced-final request is allowed only after
  an empty or recognized provider-notice length stop.
- Temperature zero and concurrency ceiling 256.
- Same compact score-array messages and strict parsers frozen before the Gemma
  V2 smoke.
- Exact same two smoke trees, conditional 20 confirmation trees,
  path-dependent hypotheses, queries, retrieved documents, external
  required-document endpoints, controls, and tie breaking as V3.1.
- No Gemma or GPT-5.4 score is shown to Qwen or reused as a response.
- No response repair, score normalization, resampling, or endpoint access
  occurs after generation.

Source hashes:

- smoke:
  `61c466ead49a5545abd54dd28e1a2fd7ad070287f5108684a3b2643befc40a01`;
- confirmation:
  `f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae`;
- frozen nonsemantic controls:
  `d1d43ecf0f09fd5874e9badd6cff4e9ae5c3da882a3ea408b816225c2d2b29f4`.

## Serving And Efficacy Smoke

Exactly 14 logical prompts: two myopic roots, two full-tree non-myopic roots,
and ten focused continuations.

Pass requires:

1. all 14 logical responses parse without repair;
2. 14--28 physical requests;
3. positive reported reasoning tokens;
4. every forced exit has exactly one successful forced-final request;
5. varying focused scores on at least 8/10 roots;
6. focused pairwise accuracy at least `.55`;
7. at least 7/10 oracle-optimal focused choices; and
8. cost at most `$0.25`.

Failure closes this exact Qwen interface without prompt, parser, reasoning
budget, score-band, threshold, task, or model-size repair.

## Conditional Confirmation

Only a passing smoke authorizes exactly 140 logical prompts and at most 280
physical requests on the 20 open confirmation trees, capped at `$2.00`.

All original V3.1 efficacy gates remain:

- non-myopic root accuracy at least `.60` and gain over fresh myopic scores at
  least `.05`;
- focused accuracy at least `.60`, optimal rate at least `.70`, mean regret at
  most `.30`, and selected-root continuation loss at most five documents;
- at least four wins, at most two losses, and total gain at least four
  documents over myopic;
- at least six wins, at most four losses, and total gain at least five over
  random;
- at least four documents gained over joint continuation selection; and
- root accuracy above `.5455`, continuation accuracy above `.6316`, and
  endpoint total at least 25 against frozen nonsemantic controls.

Passing supports cross-family semantic-scorer transfer under a smaller,
thinking-enabled, compute-unmatched model. It does not establish fresh-task
generalization or correct-belief causality.

## Budget

The smoke is projected below `$0.15` with a hard `$0.25` cap. Confirmation is
projected below `$1.50` with a hard `$2.00` cap. Live credits and the protected
reserve must be checked before each stage. OatML remains paused.
