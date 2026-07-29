# ICAE Selective Full-World Instrument Smoke V2

Date: 2026-07-29

## Purpose

Prospectively rerun the selective full-world instrument after V1 stopped on
the first response's array order. V1 returned all unique world/question
indices and probabilities summing to 100, but placed index zero last. It
remains failed and was not normalized or resumed.

V2 changes only the response codec: every collection whose rows carry
explicit semantic keys is an unordered keyed set. Parsers validate uniqueness
and complete key coverage, then canonicalize internally. This applies
prospectively to worlds, questions, likelihood cells, retention rows, and
endpoint rows. Belief semantics, gates, and call budget are unchanged.

## Frozen Task And Models

Apply the same public mechanics eligibility rule and select the task minimizing
`SHA256("51600:<alias>")`:

- `realcode@015` (Python).

This is a different task, already opened by the mechanics source audit.

Models:

- GPT-5.4 non-thinking, seed `51700`: world generation and refresh;
- GPT-5.4 Mini non-thinking, seed `51800`: likelihoods, retention, semantic
  trigger match, and duplicate endpoints.

## Exact Ten Calls And Gates

The exact ten calls and every scientific threshold are identical to V1:

1. initial eight-world/six-question belief;
2. six hypothetical positive answers;
3. 8-by-6 likelihoods;
4. positive branch refresh;
5. fallback branch refresh;
6. branch retention;
7. realized exact-controller match;
8. realized refresh;
9. per-world hidden coverage;
10. exact endpoint repeat.

Required gates remain:

- exact 10 accepted/HTTP requests with zero retries, reasoning, or forced
  exits;
- valid unique full-world supports with probability sum one and effective
  world count at least `2.0`;
- at least three likelihood values;
- both branches and actual history change at least four questions;
- branch supports differ and retention counts differ by at least one;
- realized root matches a released trigger;
- duplicate endpoint judgments are exact;
- weighted coverage is in `[0.20, 0.90]` with per-world range at least `0.10`;
- cost at most `$0.50`.

No response repair, reissue, retry, task/model/seed substitution,
development-or-later partition, or executable test endpoint is allowed.
Passage authorizes only a separately frozen paired measurement on an unopened
cohort; failure closes V2.
