# tau-Knowledge Claude Sonnet 5 Scorer Replication

## Status

Frozen before any Anthropic model response. This is a post hoc cross-model
replication on the already open tau-Knowledge V3.1 trees, not a new held-out
environment result. Required-document endpoints are known to the researchers
but remain absent from every model prompt.

## Fixed Inputs

- Serving tree artifact SHA-256:
  `61c466ead49a5545abd54dd28e1a2fd7ad070287f5108684a3b2643befc40a01`.
- Twenty-task confirmation tree artifact SHA-256:
  `f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae`.
- Frozen nonsemantic-control analysis SHA-256:
  `d1d43ecf0f09fd5874e9badd6cff4e9ae5c3da882a3ea408b816225c2d2b29f4`.
- Model: `anthropic/claude-sonnet-5` through OpenRouter.
- Temperature: `0`.
- Reasoning is not requested; any reported reasoning token fails the stage.
- No opening, information-need, root-query, followup-query, or retrieval call is
  regenerated. Claude sees the exact compact scorer inputs used by GPT-5.4.

## Calls

Serving smoke uses the two frozen V3 smoke trees and exactly 14 physical calls:

- two myopic root scorers;
- two full-tree non-myopic root scorers; and
- ten focused continuation scorers.

If every serving gate passes, confirmation uses the 20 frozen V3.1 trees and
exactly 140 physical calls:

- 20 myopic root scorers;
- 20 full-tree non-myopic root scorers; and
- 100 focused continuation scorers.

No response is retried, repaired, replaced, or manually interpreted. Raw text
is stored privately outside git.

## Parser

The semantic prompts and requested keys are unchanged. The parser is frozen
before model use and accepts an unambiguous JSON integer or decimal digit
string for every numeric field:

- root score: `0` through `100`;
- best followup: `1` through `4`;
- focused count score: `0-9`, `30-39`, `60-69`, or `90-99`.

One- or two-digit zero padding is accepted for focused scores, matching V3.1.
Root scores may use up to three digits. Parsed numeric values are identical
regardless of JSON number/string representation. Keys, ranges, rationales, and
all semantic content remain strict.

## Serving Gates

The smoke passes only if:

- exactly two cases and ten focused roots complete;
- exactly 14 physical requests and zero reasoning tokens are recorded;
- at least 8/10 focused roots have nonconstant continuation scores;
- focused pairwise accuracy is at least `0.55`; and
- at least 7/10 selected continuations are oracle-optimal.

Failure stops before the 20-task replication.

## Confirmation Gates

All original V3.1 confirmation gates remain binding after replacing only the
scorer model:

- root comparable pairs at least 50;
- non-myopic root pairwise accuracy at least `0.60`;
- non-myopic-minus-myopic root accuracy at least `0.05`;
- continuation comparable pairs at least 200;
- focused continuation accuracy at least `0.60`;
- oracle-optimal continuations at least 70/100;
- mean continuation regret at most `0.30`;
- selected-root continuation loss at most 5;
- end-to-end versus myopic: at least 4 wins, at most 2 losses, total gain at
  least 4;
- end-to-end versus seeded random: at least 6 wins, at most 4 losses, total
  gain at least 5; and
- focused improvement over the original joint selector at least 4 documents.

Three additional transfer checks compare Claude against the strongest frozen
nonsemantic controls on the identical trees:

- root pairwise accuracy must exceed `0.5455`;
- focused continuation accuracy must exceed `0.6316`; and
- exact end-to-end coverage must be at least 25 documents, three above the
  strongest nonsemantic endpoint of 22.

Passing supports scorer-level cross-model robustness. It does not establish a
second held-out policy result because tree generation and endpoints were
already open.

## Budget

Before model use, OpenRouter reports `$52.069693036` remaining, or
`$27.069693036` above the protected `$25` reserve. Estimated prompt cost is
about `$0.12` for smoke and `$1.17` for confirmation before completions.
Smoke is capped at `$0.50`; confirmation at `$3.00`. OatML remains paused.
