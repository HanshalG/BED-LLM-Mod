# InteractComp Model-Criticism Validation V3 Amendment

Date: 2026-07-25

## Status

Frozen before any V3 response. This incorporates the V1 scientific
preregistration and V2 classification-format amendment. V3 changes only the
fresh task block, seeds, and multilingual terminal question mark.

## V2 Failure

V2 enrolled six collapsed supports and generated all 24 roots. Four clean,
single-line Chinese questions ended with the standard full-width question mark
`？`; the inherited parser accepted only ASCII `?`. V2 stopped after 152 Mini
calls before any classification, auxiliary support, responder, context, target,
score, or endpoint.

The returned questions directly establish that this is a punctuation interface
issue. No paid serving smoke or V2-task reuse is needed.

## Fresh V3 Block

- Validation seed: `24384`.
- Random-control seed: `24385`.
- Next 16 untouched manifest indices:
  `86, 169, 41, 28, 85, 103, 66, 145, 138, 192, 156, 114, 14, 44, 162, 63`.
- Corresponding benchmark IDs:
  `87, 170, 42, 29, 86, 104, 67, 146, 139, 193, 157, 115, 15, 45, 163, 64`.

V1 and V2 tasks and responses are excluded. V3 enrolls the first six tasks with
at most four exact normalized unique entities among eight initial particles.

## Question Amendment

A root question is valid only when:

1. it contains exactly one nonempty line;
2. its stripped length is between 8 and 300 characters, inclusive; and
3. its final character is exactly ASCII `?` or full-width `？`.

No other punctuation, prefix, suffix, line joining, extraction, rewriting, or
translation is accepted.

The V2 classification rule remains unchanged: remove only ASCII space, tab,
carriage return, and newline, then require exactly four Y/N/U characters.

## Unchanged Protocol And Gates

All scientific elements remain unchanged:

- non-thinking GPT-5.4 Mini generator, semantic validator, and classifier;
- non-thinking GPT-5.4 context-only responder;
- four roots, 16 outside proposals, first eight semantically distinct;
- primary balanced model-identity MI;
- ordinary current EIG, compute-matched augmented-support EIG, and seeded random
  controls;
- eight realized refresh particles per root;
- delayed exact-target-mass endpoint;
- exact 632 Mini and 24 responder calls, 656 total;
- all 15 V1 scientific/integrity gates; and
- `$1.50` hard cap, no repair, replacement, or same-interface rerun.

## Budget

- Projected cost: `$0.50`.
- Hard cap: `$1.50`.
- Project-ledger spend before V3: `$86.47568266920753`.
- Monday local allowance remaining: `$14.667622149999886`.
- Authenticated OpenRouter remaining: `$43.927137884`, or `$18.927137884`
  above the protected `$25` reserve.
- OatML resources: prohibited.

## Verification

The full V3 fixture completed exactly 656 simulated calls with all serving and
integrity gates passing; target-based gates failed on target-free fixtures as
intended. Focused tests:

```text
pytest -q tests/test_interactcomp_model_criticism_validation.py \
  tests/test_interactcomp_semantic_path_smoke.py \
  tests/test_interactcomp_robust_support_development.py \
  tests/test_interactcomp_first_link_opportunity.py \
  tests/test_helpers_load_config.py tests/test_core_config.py
115 passed
```
