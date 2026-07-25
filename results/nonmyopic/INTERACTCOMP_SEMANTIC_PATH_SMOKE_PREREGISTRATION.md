# InteractComp V2 Semantic-Path Smoke Preregistration

Date: 2026-07-25

## Purpose

Test the three serving stages that V1 never reached before implementing a fresh
model-criticism validation V2:

1. outside-support particle generation;
2. semantic same/distinct validation; and
3. auxiliary-particle Y/N/U classification.

This is a target-free serving smoke on an already-open task. It has no
scientific endpoint.

## Frozen Inputs And Interface

- Reuse only already-open task index `141`, benchmark ID `142`.
- Reuse current particles and four roots from the V1 private checkpoint,
  SHA-256
  `691130ba00f29936ba1bb250b84b6e7a0e5692dd62e48c94ec80c5cf3c696c06`.
- Decrypt only the task's ambiguous question.
- Do not decrypt or prompt with hidden context or exact target answer.
- Model: `openai/gpt-5.4-mini`, non-thinking.
- Generate four outside-support candidates at temperature `.7`.
- Judge each candidate `D` or `S` relative to current support at temperature
  `0`.
- Classify all four candidates on all four existing roots at temperature `0`.

V2's only grammar change is prospectively frozen here: remove ASCII space, tab,
carriage return, and newline characters, then require exactly four Y/N/U
characters. Commas, prefixes, prose, or any other character remain invalid.

## Exact Calls And Gates

Exactly 12 calls:

- four outside-support proposals;
- four semantic validations; and
- four classifications.

Every gate must pass:

1. exactly 12 physical requests and 12 HTTP attempts;
2. zero retries, reasoning tokens, and forced exits;
3. all four proposals parse under the existing strict two-line grammar;
4. every semantic judgment parses as exactly `D` or `S`;
5. at least two proposals are judged semantically distinct;
6. all four classifications parse after ASCII-whitespace compaction; and
7. cost is at most `$0.15`.

There is no repair, replacement, response reissue, context access, target
access, or endpoint. Passage authorizes implementation and preregistration of
V2 on the next 16 untouched manifest tasks. Failure closes this semantic path.

## Budget

- Projected cost: `$0.03`.
- Hard cap: `$0.15`.
- Project-ledger spend before run: `$86.37242356920763`.
- Monday local allowance remaining: `$14.770881249999789`.
- Last authenticated live balance: `$44.111601884`, or `$19.111601884` above
  the protected `$25` reserve.
- OatML resources: prohibited.

## Deterministic Verification

The fixture passed every gate with exactly 12 simulated calls, including
whitespace-separated labels, and loaded no context or target. Focused tests:

```text
pytest -q tests/test_interactcomp_semantic_path_smoke.py \
  tests/test_interactcomp_model_criticism_validation.py \
  tests/test_interactcomp_robust_support_development.py \
  tests/test_interactcomp_first_link_opportunity.py
12 passed
```
