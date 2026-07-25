# Ambig-IaC V2 Particle-Schema Serving Smoke Preregistration

Date: 2026-07-25

## Purpose

Test whether GPT-5.4 Mini can serve the distinct unambiguous V2 particle format
exactly before any fresh Ambig-IaC task is released for scientific efficacy.
This is a schema-only mechanics test. It cannot support a non-myopic claim.

## Frozen Inputs

- Same already-open Ambig-IaC prompts from V1: tasks `272`, `66`, `156`.
- Official source commit:
  `4b50c142ed4638a4caaee9ca0b92d8d0e5b8c8cb`.
- Dataset tree SHA-256:
  `1f2d20cadb147962fac104e037d75c53d68d1a9b092050a80513d3751c683f63`.
- No target `plan.json` file may be opened.
- No branch, score, answer, or endpoint is computed.

## Distinct V2 Interface

- Model: `openai/gpt-5.4-mini` through OpenRouter.
- Reasoning disabled; temperature `.7`; maximum output 900 tokens.
- Exactly one independent response per open task.
- The response is exactly one JSON object with exactly:
  `resources`, `dependencies`, and `attribute_keys`.
- `resources` is a nonempty array of exact
  `{"label": string, "address": string}` objects.
- `dependencies` is an array of exact
  `{"source": label, "depends_on": label}` objects.
- `attribute_keys` is an array of exact
  `{"label": label, "keys": [nonempty strings]}` objects.
- Nested objects in string fields, unknown labels, self-dependencies, duplicate
  labels, malformed JSON, prose, and extra keys are invalid.
- Valid V2 objects are converted deterministically to the same canonical
  resource/topology/attribute proposition representation used by V1.
- No normalization, repair, first-object extraction, response replacement, or
  reissue is allowed.

## Exact Contract And Gates

- Physical requests: exactly 3.
- HTTP attempts: exactly 3.
- Transport retries: zero.
- Reasoning tokens: zero.
- Forced exits: zero.
- All three responses pass the exact V2 parser.
- Every valid specification yields at least one proposition feature.
- Cost is at most `$0.15`.

All gates must pass. A pass authorizes only a separately frozen V2 efficacy
smoke on untouched seeded tasks. A failure closes this V2 schema and Ambig-IaC
for the current project; there will be no V3 serving amendment.

## Budget

The run is projected at `$0.03` and has a hard `$0.15` run cap. Immediately
before preregistration, project-ledger spend was `$86.17832606920743`, leaving
`$14.96497875` under the Monday new-work ceiling. The authenticated OpenRouter
balance was `$44.206476634`, leaving `$19.206476634` above the protected `$25`
reserve. Both must still pass immediately before the paid command. OatML is not
used.

## Zero-Call Verification

The exact entry point passed a deterministic local dry run with three simulated
requests, three valid V2 objects, all gates true, strict JSON artifacts, and
zero cost. Focused tests:

```text
pytest -q tests/test_ambig_iac_schema_v2_smoke.py \
  tests/test_ambig_iac_first_link_smoke.py
10 passed
```
