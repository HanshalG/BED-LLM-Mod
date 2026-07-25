# InteractComp Model-Criticism Validation V2 Amendment

Date: 2026-07-25

## Status

This amendment is frozen before any V2 response. It incorporates the complete
V1 preregistration and changes only the fresh task block, seeds, and
classification whitespace grammar described below.

## V1 Failure And Serving Evidence

V1 stopped before auxiliary generation or endpoint access because one current
classification was `YYY Y`, while the frozen parser required exact adjacent
characters.

The target-free semantic-path smoke subsequently passed all gates on an
already-open task:

- exact 12/12 Mini calls;
- four valid outside-support proposals;
- four valid semantic judgments, all distinct;
- four valid auxiliary classifications;
- zero retries, reasoning, or forced exits; and
- no context, target, refresh, score, or endpoint.

No V1 screened task or response is reused in V2.

## Fresh V2 Block

- Validation seed: `24382`.
- Random-control seed: `24383`.
- Next 16 untouched manifest indices:
  `90, 12, 133, 130, 184, 125, 105, 69, 67, 124, 11, 64, 91, 9, 60, 197`.
- Corresponding benchmark IDs:
  `91, 13, 134, 131, 185, 126, 106, 70, 68, 125, 12, 65, 92, 10, 61, 198`.

The task contents have not been inspected. V2 prospectively enrolls the first
six tasks with at most four exact normalized unique entities among eight
initial particles, exactly as V1 did.

## Format Amendment

For both current and auxiliary Y/N/U classifications:

1. remove only ASCII space, tab, carriage return, and newline characters;
2. require the remaining string to be exactly four characters; and
3. require every character to be `Y`, `N`, or `U`.

Commas, labels, prefixes, explanations, punctuation, Unicode whitespace, wrong
length, or any other character fail. No semantic normalization or response
repair is introduced.

## Unchanged Scientific Protocol

Everything else is unchanged from
`INTERACTCOMP_MODEL_CRITICISM_VALIDATION_PREREGISTRATION.md`:

- `openai/gpt-5.4-mini` non-thinking generator, semantic validator, and
  classifier;
- `openai/gpt-5.4` non-thinking context-only responder;
- four roots per enrolled task;
- 16 outside proposals, retain first eight semantically distinct;
- primary balanced current-vs-auxiliary model-identity MI;
- paired ordinary EIG, compute-matched augmented-support EIG, and seeded random
  controls;
- eight realized refreshed particles per root;
- exact target-answer mass endpoint;
- target answers loaded only after all calls and scores freeze;
- exact 632 Mini plus 24 responder calls, 656 total;
- every original scientific and integrity gate unchanged; and
- `$1.50` hard cap with no repair, replacement, or same-interface rerun.

## Budget

- Projected cost: `$0.50`.
- Hard cap: `$1.50`.
- Project-ledger spend before V2: `$86.37976006920765`.
- Monday local allowance remaining: `$14.763544749999767`.
- Authenticated OpenRouter remaining: `$44.005042634`, or `$19.005042634`
  above the protected `$25` reserve.
- OatML resources: prohibited.

## Verification

The exact V2 fixture completed 656 simulated calls, enrolled six collapsed
supports, passed every serving/integrity gate, and failed target-based gates on
target-free fixture particles as intended. Focused tests:

```text
pytest -q tests/test_interactcomp_model_criticism_validation.py \
  tests/test_interactcomp_semantic_path_smoke.py \
  tests/test_interactcomp_robust_support_development.py \
  tests/test_interactcomp_first_link_opportunity.py \
  tests/test_helpers_load_config.py tests/test_core_config.py
114 passed
```
