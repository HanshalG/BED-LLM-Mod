# SWE-Interact Native GPT-5.5 Mechanics Preregistration

Date: 2026-07-28

**Status: frozen before any GPT-5.5 SWE-Interact request.**

## Why V2 Is Distinct

V1 used GPT-5.4 as the released-user role. It failed formally because the
independent judge inconsistently included the `R` prefix, and a zero-call
diagnostic showed that GPT-5.4 also disclosed most or all hidden requirements
to a generic checklist request.

The official SWE-Interact multi-turn run configuration specifies
`openai/gpt-5.5` with high reasoning. OpenRouter currently exposes that exact
model with seed and reasoning-effort support. V2 tests whether V1's
anti-extraction failure was caused by substituting GPT-5.4 for the released
simulator model.

V2 is not a V1 rescue:

- all 27 simulator/judge requests are fresh;
- no V1 response is replayed or scored as V2 evidence;
- the generic anti-extraction threshold is unchanged;
- development, confirmation, and retained values remain sealed.

## Frozen Source, Tasks, And Branches

Source commit, tree, manifest, and the three one-per-family mechanics tasks are
identical to V1. Each task again receives one fresh initial reply followed by
seven forked branches from that exact history:

1. generic complete-checklist request;
2. targeted root A twice;
3. targeted root B twice;
4. review surface A;
5. review surface B.

The root text, review-snapshot text, intended requirement sets, and semantic
gates are byte-identical to V1.

## Models And Interface

- Released-user role: `openai/gpt-5.5`, high reasoning, temperature zero.
- Independent annotator: `openai/gpt-5.4-mini`, reasoning disabled,
  temperature zero.
- Seed: `24424`.
- No BED policy is present. Later BED policies remain nonreasoning; only the
  naive baseline may think.

The only transport repair is prospective and exact: the judge must emit bare
decimal requirement indexes, never an `R` prefix:

```text
INITIAL|1
GENERIC|NONE
ROOT_A_1|2,4
```

Rows are order-insensitive through the core keyed-row codec. The parser rejects
prefixes, leading zeros, signs, decimals, duplicates, unknown indexes, missing
keys, empty values, extra fields, and prose. No repair, reissue, fallback, or
post-parse coercion is allowed.

## Accounting

- GPT-5.5 user calls: `24`
- GPT-5.4 Mini judge calls: `3`
- Total logical calls: `27`
- Concurrency: at most `24`
- Maximum transport retries: `3` per adapter, separately counted
- User output ceiling: `4,096` tokens
- Judge output ceiling: `1,024` tokens
- Projected cost: `$1.00`
- Runaway stage ceiling: `$4.00`

The ceiling is not a reserve. The full provider balance remains available by
expected scientific value.

## Frozen Gates

All V1 integrity and semantic gates remain conjunctive:

1. exact 24 user and 3 judge requests;
2. HTTP attempts equal requests plus bounded transport retries;
3. no empty output, forced exit, or semantic repair;
4. zero judge reasoning tokens;
5. combined cost at most `$4`;
6. exactly eight valid numeric keyed rows per task;
7. initial reply has at most one concrete requirement on `3/3` tasks;
8. generic branch has zero new requirements on `3/3` tasks;
9. both targeted roots disclose intended, distinct requirements on `3/3`;
10. exact repeated-root annotation agreement on `6/6` pairs;
11. both review surfaces disclose intended, distinct corrections on `3/3`.

## Decision

A pass authorizes a separately frozen first-link mechanics protocol using
nonreasoning LLM-generated semantic hypotheses and actions. A failed generic
gate closes SWE-Interact as a non-myopic clarification route under its exact
released simulator model. Any other failure closes V2 without repair or rerun.
