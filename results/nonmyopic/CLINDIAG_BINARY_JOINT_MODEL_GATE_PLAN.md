# ClinDiag Binary Joint-Model Gate

Date: 2026-07-24

Status: **preregistered before any response from this interface.**

## Distinct Construction

The preceding four-way smoke showed a promising semantic likelihood but an invalid
hidden environment. Several target-blind option sets did not contain the hidden
patient's actual result, so the gatekeeper selected or synthesized an inconsistent
closest option.

This new construction asks only specific binary clinical propositions. Every action
has an exhaustive `yes`/`no` answer space, while the gatekeeper also returns a short
objective finding. The query parser rejects invasive confirmatory and treatment
actions. GPT-5.4 non-reasoning replaces GPT-5.4 Mini as the hidden environment and
judge. This is a new interface/model qualification, not a repair or reinterpretation
of the failed four-way gate.

## Frozen Cases

Selection seed `24292` fixes six fresh cases, all disjoint from prior ClinDiag use and
the sealed holdout.

Interface smoke:

- `17983370`;
- `rare231`.

Joint-model smoke, used only if the interface smoke passes:

- `15492352`;
- `24867998`;
- `rare264`;
- `rare173`.

## Ten-Call Interface Smoke

For each of two cases:

1. generate a 12-diagnosis support from only the initial presentation;
2. generate six target-blind binary clinical propositions;
3. ask the hidden gatekeeper propositions `q1` and `q2`;
4. repeat `q1` with an exact identical prompt.

Expected calls: `2 + 2 + 4 + 2 = 10`.

Pass requires:

- exactly 10 calls and zero reasoning;
- both supports parse at size 12;
- both six-query sets pass the fixed action parser;
- all six gatekeeper responses parse, contain no literal target leak, and never
  report missing/unavailable source data;
- both exact duplicates return the same answer.

No retries are allowed. Failure closes the binary line before the larger smoke.

## Frozen Joint-Model Smoke

For each of four fresh cases:

- one 12-diagnosis support;
- six target-blind binary propositions;
- six hidden-case answers;
- one exact duplicate of `q1`;
- one likelihood matrix for six generated hypotheses plus the true diagnosis added
  only after queries and realized outcomes are frozen;
- one independent behavioral audit.

Expected calls:

| Stage | Calls |
|---|---:|
| Initial supports | 4 |
| Binary query sets | 4 |
| Original gatekeeper answers | 24 |
| Exact gatekeeper duplicates | 4 |
| Likelihood matrices | 4 |
| Independent audits | 4 |
| **Total** | **44** |

All models are GPT-5.4 with reasoning disabled. Temperature is zero except the
already-qualified initial support sampler. No retries are allowed.

## Fixed Action Grammar

Each query must:

- have ID `q1` through `q6`;
- be one history, examination, or test proposition answerable yes/no;
- request exactly one observable fact;
- define objective positive and negative findings;
- avoid diagnosis claims, interpretation, treatment, and bundled panels;
- avoid biopsy, pathology, histology, genetics, sequencing, molecular testing,
  surgery, resection, implantation, transfusion, and other invasive confirmatory
  procedures.

## Frozen Joint Gates

All must pass:

1. exactly 44 calls and zero reasoning;
2. all supports, query sets, answers, likelihood rows, and audits parse;
3. all 24 answers pass relevance, objectivity, no-target-leak, and case-consistency
   audit;
4. all four duplicate answers and semantic findings agree;
5. every likelihood is finite and in `[0,1]`;
6. mean probability of the realized answer under the true diagnosis is at least
   `.65`;
7. the true diagnosis assigns more probability to the realized answer than the mean
   of the six generated hypotheses on at least 12/24 queries;
8. mean true-minus-generated realized-answer probability is at least `+.10`.

Passing authorizes only a fresh structural-opportunity gate with replicated
path-dependent support refresh. It does not authorize a policy or holdout directly.
Failure closes this exact binary prompt/model interface without threshold repair.
