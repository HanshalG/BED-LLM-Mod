# SymptomCheck dynamic-support source protocol

Date frozen: 2026-08-13

Status: **value-blind source audit only; zero model calls**

## Scientific target

Use SymptomCheck Bench's Avey vignettes as latent semantic worlds for adaptive
diagnostic questioning. The released doctor prompt receives only demographics;
the released patient prompt is grounded in the private presentation, chief
complaints, absent findings, physical history, family history, and social
history. The diagnosis remains an endpoint. This offers a cleaner released
separation than the closed AgentClinic NEJM interface and a population whose
physical file can be bound directly.

If all prerequisites pass, the desired first link is depth-two planning over
answer-conditioned regenerated diagnostic support against a compute-matched
myopic ensemble, fixed-support, history-blind, and random controls. Source shape
alone authorizes no model call or efficacy claim.

## Immutable source

- repository: `https://github.com/medaks/medask-benchmarks`
- commit: `36c80b44fee77da271d97d38c43e49926dc95b8e`
- tree: `b0649fb98bb150889974974dd8737b4ced30ee58`
- license: MIT
- data: `symptomcheck_bench/vignettes/avey_vignettes.jsonl`
- data SHA-256:
  `f516ee0fb17bdaefca7f53483fb4e38a1eb3c72cf52cf45d34dcf0484bf66976`
- expected nonblank population: `400`
- simulator SHA-256:
  `5400d243eb749b49335cde4c971303f68c8479c94a04c057a9f08c37699a22a7`
- vignette adapter SHA-256:
  `0721796828a3efc8f198466a1d433e0643928173d366815e4527908e411ad7c5`
- agent prompts SHA-256:
  `117da867d1056612cde29918e3033c95b82573fe4868a5b953cc6f3559a75867`
- benchmark entrypoint SHA-256:
  `622461ee9bfc2f84ca559f34501b499a2c981b09d901961a7443be002dfa24e2`

## Value-blind gates

The source passes only if every gate holds:

1. Repository, commit, tree, and all bound hashes match exactly.
2. The data contains exactly 400 nonblank JSON objects.
3. Every row has the same exact eight-field schema used by the released Avey
   adapter: `correct_diagnosis`, `demographics`, `presentation`,
   `chief_complaints`, `absent_findings`, `physical_history`,
   `family_history`, and `social_history`.
4. Canonical rows are unique and every required field is structurally nonempty.
5. The population contains at least 100 distinct normalized diagnoses.
6. For every row, policy-visible demographics are not canonically identical to
   any private patient field or the diagnosis.
7. The released doctor prompt receives demographics only; the patient prompt
   receives all seven nondiagnostic fields, reveals only relevant requested
   information, and returns `I don't know` for absent information.
8. The released simulator maintains separate doctor and patient histories,
   alternates patient responses and doctor questions, enforces a finite dialogue
   horizon, and exposes diagnosis only through the endpoint accessor.
9. The deterministic value-blind split is complete and disjoint.

Case identity is SHA-256 of canonical compact row JSON. Cases are ordered by
`SHA256("symptomcheck-dynamic-support-v1|" + case_id)` and allocated:

| Split | Count | Access after source audit |
|---|---:|---|
| mechanics | 6 | only after a passing source audit and pushed protocol |
| opportunity | 30 | sealed |
| development | 64 | sealed |
| confirmation | 96 | sealed |
| reserve | 204 | sealed |

Public artifacts serialize only aggregate counts, booleans, exact schema names,
and hashes of complete ordered case-ID lists. They serialize no individual case
ID, demographics, complaint, history, finding, diagnosis, dialogue, or endpoint.

## Required mechanics successor

A source pass authorizes only a separately frozen six-case mechanics gate. It
must establish before diagnosis access:

- a strict atomic question grammar and finite candidate set;
- patient answer obedience, missing-information behavior, and repeated-answer
  stability under exact prompts and seeds;
- no diagnosis leakage into patient or policy prompts;
- at least four generated diagnostic hypotheses with nontrivial truth coverage;
- answer-conditioned support regeneration that changes truth coverage;
- calibrated semantic response likelihoods under positive and negative answers;
- at least two dependent information actions before diagnosis;
- a depth-two root that differs from the compute-matched myopic ensemble and
  has positive oracle-linked first-link value on at least four of six cases;
- paired common-random-number diagnosis endpoints and random control;
- complete candidate-score banking before correct diagnoses are opened.

Any source or mechanics failure closes this exact construction. There is no case
replacement, subset repair, threshold relaxation, or endpoint-informed prompt
repair.

## Accounting

- OpenRouter calls: `0`
- OpenRouter cost: `$0`
- OATML cluster use: none
