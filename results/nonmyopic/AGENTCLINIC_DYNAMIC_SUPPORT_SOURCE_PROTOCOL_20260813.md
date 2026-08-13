# AgentClinic dynamic-support source protocol

Date frozen: 2026-08-13

Status: **source audit only; zero model calls**

## Scientific target

Use AgentClinic's released NEJM cases as semantic latent worlds for a two-question
diagnostic BED task. A fixed, policy-independent chief-complaint intake precedes
the comparison. The LLM must generate and regenerate the diagnostic hypothesis
support and predict semantic patient/test responses. The eventual policy
comparison is dynamic-support depth two versus a compute-matched myopic scorer,
with fixed-support, history-shuffled, and random controls.

This is not authorized by source shape alone. In particular, a clinical dialogue
is not automatically a non-myopic BED problem.

## Immutable source

- repository: `https://github.com/SamuelSchmidgall/AgentClinic`
- commit: `b6570edefb940857a7c334350656b29f9d984f24`
- tree: `084556771761828ac0fd121d37353207de91486c`
- data: `agentclinic_nejm_extended.jsonl`
- data SHA-256: `d945305ee17ee1456053fbfe2e9d9c5e8b27d14538bf48ab8ace7306dc437b85`

The source audit may inspect row shape and value presence, but it must serialize
no question, patient history, examination finding, image URL, answer text, or
diagnosis. It must not invoke AgentClinic agents or any model.

## Population gates

The release passes only if all of the following hold:

1. Exactly 120 rows have the exact six-field schema released by AgentClinic.
2. Canonical rows are unique.
3. Every row has exactly one marked-correct answer.
4. Every question, image URL, patient history, physical-exam record, and type
   field is nonempty.
5. Every image URL uses HTTPS.
6. The deterministic value-blind partition is complete and disjoint.

Case identity is the SHA-256 of canonical compact JSON. Cases are ordered by
`SHA256("agentclinic-dynamic-support-v1|" + case_id)` and allocated as:

| Split | Count | Access after this audit |
|---|---:|---|
| mechanics | 6 | only after a passing source audit |
| opportunity | 24 | sealed |
| development | 30 | sealed |
| confirmation | 40 | sealed |
| reserve | 20 | sealed |

Public artifacts contain only counts and hashes of ordered case-ID lists. They
contain no individual case ID or source value.

## Mechanics gate required next

Only the six mechanics cases may be opened after this source boundary is pushed.
Before any paid call, freeze an exact mechanics protocol that requires:

- a deterministic fixed intake shared by all policies;
- strict doctor-action grammar separating patient questions from test requests;
- target patient/test responses that never reveal the gold diagnosis directly;
- exact schema, answer-obedience, repeated-call stability, and missing-information
  behavior on a serving cohort;
- generated differential support with nontrivial truth coverage and at least
  three viable diagnoses;
- branch-conditioned support regeneration with a measurable truth-coverage
  change;
- a changed depth-two root and positive oracle-linked first-link value versus
  compute-matched myopic on mechanics cases;
- no diagnosis endpoint access until all candidate scores are frozen.

Failure closes this exact source construction. No case replacement, split
relaxation, threshold change, or post-outcome prompt repair is allowed.

## Accounting

- OpenRouter calls: `0`
- OpenRouter cost: `$0`
- OATML cluster use: none
