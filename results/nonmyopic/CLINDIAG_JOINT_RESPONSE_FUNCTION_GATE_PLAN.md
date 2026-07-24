# ClinDiag Joint Patient-Response Function Gate

Date: 2026-07-24

Status: **preregistered before any response from this interface.**

## Motivation

The independent binary gatekeeper exposed chart missingness for an absent fact and
changed provenance across an exact duplicate. This construction instead asks the
hidden environment to instantiate answers to all six already-frozen target-blind
binary propositions jointly. The complete response function is duplicated and audited
as one object, encouraging cross-query consistency and explicit synthesis of every
unrecorded fact.

The binary query grammar and blocked invasive-action parser remain frozen from commit
`52342f5`. A new response parser, fixed before serving, additionally rejects “not
reported,” “no documentation,” “absent from the record,” and equivalent missingness.

## Fresh Split

Selection seed `24293` fixes six unused cases, disjoint from every earlier ClinDiag
case and the sealed holdout.

Ten-call qualification:

- `24450895`;
- `rare214`.

Likelihood-fidelity smoke, touched only if qualification passes:

- `25688156`;
- `12830418`;
- `rare43`;
- `rare88`.

## Ten-Call Qualification

Per case:

1. generate a 12-diagnosis support from the initial presentation;
2. generate six safe target-blind binary propositions;
3. generate all six hidden-patient answers jointly;
4. repeat the complete answer-function prompt exactly;
5. independently audit both answer functions.

Expected calls: `2 + 2 + 2 + 2 + 2 = 10`.

Pass requires:

- exactly 10 calls, zero reasoning, and zero retries;
- both supports and query sets pass their frozen parsers;
- all 24 original-plus-duplicate answers parse without missingness or literal target
  leakage;
- original and duplicate agree on all 12 yes/no labels;
- an independent GPT-5.4 audit accepts relevance, objectivity, no target leak,
  case consistency, semantic duplicate consistency, and provenance consistency for
  all 12 query pairs.

Failure closes this exact response-function interface before likelihood evaluation.

## Conditional Likelihood Smoke

If qualification passes, four fresh cases receive the same stages plus one binary
likelihood matrix over six generated hypotheses and the true diagnosis, added only
after actions and realized answers are frozen.

Expected calls:

| Stage | Calls |
|---|---:|
| Supports | 4 |
| Query sets | 4 |
| Joint answer functions | 4 |
| Exact full-function duplicates | 4 |
| Likelihood matrices | 4 |
| Independent audits | 4 |
| **Total** | **24** |

All roles use GPT-5.4 without reasoning. Pass additionally requires:

- mean probability of the realized answer under truth at least `.65`;
- truth exceeds the six generated hypotheses on at least 12/24 queries;
- mean truth-minus-generated realized-answer probability at least `+.10`.

Passing authorizes only a fresh replicated structural-opportunity gate. No planner or
holdout is authorized directly.
