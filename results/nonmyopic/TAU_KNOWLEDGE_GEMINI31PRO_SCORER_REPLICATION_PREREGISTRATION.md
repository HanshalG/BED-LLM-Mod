# tau-Knowledge Gemini 3.1 Pro Scorer Replication

## Status

Frozen before any Gemini response. This is a post hoc cross-model replication
on the already open tau-Knowledge V3.1 trees, not a new held-out environment
result. Required-document endpoints remain absent from every model prompt.

## Fixed Inputs

- Serving tree artifact SHA-256:
  `61c466ead49a5545abd54dd28e1a2fd7ad070287f5108684a3b2643befc40a01`.
- Twenty-task confirmation artifact SHA-256:
  `f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae`.
- Frozen nonsemantic analysis SHA-256:
  `d1d43ecf0f09fd5874e9badd6cff4e9ae5c3da882a3ea408b816225c2d2b29f4`.
- Model: `google/gemini-3.1-pro-preview` through OpenRouter.
- Temperature: `0`.
- Maximum output: 4,096 tokens.
- Reasoning is not requested; any reported reasoning token fails the stage.
- No query, belief, tree, retrieval, or endpoint is regenerated.
- Gemini sees the exact compact semantic scorer inputs used by GPT-5.4.

This is distinct from the closed Claude serving attempt. Claude spent a
2,048-token allowance in hidden reasoning and returned empty full-tree answers.
Gemini advertises explicit reasoning controls and receives a preregistered
4,096-token visible-output allowance; no Claude response is reused.

## Calls

Serving uses two frozen smoke trees and exactly 14 calls: two myopic root
scorers, two non-myopic root scorers, and ten focused continuation scorers.
Only an all-gates pass unlocks the exact 140-call confirmation on 20 trees.

No response is semantically retried, repaired, replaced, or manually
interpreted. Raw responses are stored privately outside git.

## Parser

The semantic prompts and required keys are unchanged. Frozen parsing accepts
unambiguous JSON integers or decimal digit strings:

- root scores from 0 through 100;
- best followups from 1 through 4; and
- focused scores in `0-9`, `30-39`, `60-69`, or `90-99`.

One- or two-digit zero padding is accepted for focused scores, as in V3.1.
Keys, ranges, nonempty root rationales, and all semantic content remain strict.

## Serving Gates

- exactly two cases, ten focused roots, and 14 physical requests;
- zero reasoning tokens;
- nonconstant focused scores on at least 8/10 roots;
- focused pairwise accuracy at least `.55`; and
- at least 7/10 selected continuations oracle-optimal.

Any failure stops before confirmation.

## Confirmation Gates

All original V3.1 gates remain:

- at least 50 comparable root pairs;
- non-myopic root accuracy at least `.60` and gain over myopic at least `.05`;
- at least 200 continuation pairs, accuracy at least `.60`, and at least 70/100
  oracle-optimal continuations;
- mean regret at most `.30` and selected-root continuation loss at most 5;
- end-to-end at least 4 wins, at most 2 losses, and total gain at least 4 over
  myopic;
- at least 6 wins, at most 4 losses, and total gain at least 5 over random; and
- at least 4 documents improvement over the original joint selector.

The transfer must also exceed the strongest frozen nonsemantic controls:

- root pairwise accuracy greater than `.5455`;
- continuation pairwise accuracy greater than `.6316`; and
- endpoint coverage at least 25 documents.

Passing supports scorer-level cross-family transfer on open trees. It does not
repair the refreshed-belief alignment null or constitute a second held-out
policy test.

## Budget

Smoke is capped at `$0.75`; confirmation at `$4.00`. Before this protocol,
OpenRouter reports `$50.578221036` remaining, or `$25.578221036` above the
protected `$25` reserve. Estimated total cost is about `$1.4` to `$2.0`.
OatML remains paused.
