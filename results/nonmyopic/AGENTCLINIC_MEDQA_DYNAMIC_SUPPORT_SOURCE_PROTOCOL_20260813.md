# AgentClinic MedQA dynamic-support source protocol

Date frozen: 2026-08-13

Status: **value-blind source audit only; zero model calls**

## Scientific target

Use AgentClinic's released extended MedQA cases as latent clinical worlds for a
two-action diagnostic BED task. Unlike the closed NEJM V1 construction, the
released MedQA interface has a policy-visible `Objective_for_Doctor` distinct
from private `Patient_Actor`, `Physical_Examination_Findings`, `Test_Results`,
and `Correct_Diagnosis` fields. This makes patient questions and test requests
potentially informative after the fixed intake.

The eventual comparison, if every prerequisite passes, is depth-two planning
over answer-conditioned regenerated diagnostic support against compute-matched
myopic, fixed-support, history-blind, and random controls. Source shape alone
does not authorize that comparison or any model call.

## Immutable source

- repository: `https://github.com/SamuelSchmidgall/AgentClinic`
- commit: `b6570edefb940857a7c334350656b29f9d984f24`
- tree: `084556771761828ac0fd121d37353207de91486c`
- code file: `agentclinic.py`
- code SHA-256:
  `ee9cfb3020c7addf717ba8eb3510ba81b2b4b53071cba67d0663ad6c7c6ddbd3`
- data file: `agentclinic_medqa_extended.jsonl`
- data SHA-256:
  `54a024eb2705c6c55d1988766adf4ab02ea7bbe2a28f843107b740032200f232`
- physical nonblank row count: `213`

The repository README describes the extended set as 215 cases, while the pinned
file contains 213 nonblank JSONL rows. This protocol binds the physical immutable
file and does not infer or manufacture two missing cases.

## Value-blind population gates

The source passes only if all of the following hold:

1. The repository, commit, tree, code hash, data hash, and 213-row physical
   population match this protocol.
2. Every row has the same exact top-level schema.
3. Every row contains an `OSCE_Examination` object with the exact five fields
   used by the released `ScenarioMedQAExtended` class:
   `Patient_Actor`, `Objective_for_Doctor`, `Physical_Examination_Findings`,
   `Test_Results`, and `Correct_Diagnosis`.
4. Canonical rows are unique.
5. All five native fields are structurally nonempty after recursive
   normalization.
6. The fixed intake is exactly the normalized `Objective_for_Doctor`; it is not
   canonically identical to `Patient_Actor`, physical findings, test results, or
   the correct diagnosis in any case.
7. A normalized scalar leaf of `Correct_Diagnosis` does not occur verbatim in
   any normalized scalar leaf of `Objective_for_Doctor`.
8. The released code independently maps policy presentation to
   `examiner_information()`, patient responses to `patient_information()`, test
   responses to `exam_information()`, and endpoints to
   `diagnosis_information()`.
9. The deterministic value-blind split is complete and disjoint.

Case identity is SHA-256 of canonical compact JSON. Cases are ordered by
`SHA256("agentclinic-medqa-dynamic-support-v1|" + case_id)` and allocated:

| Split | Count | Access after source audit |
|---|---:|---|
| mechanics | 6 | only after a passing source audit and pushed protocol |
| opportunity | 32 | sealed |
| development | 50 | sealed |
| confirmation | 80 | sealed |
| reserve | 45 | sealed |

Public artifacts serialize only aggregate counts, booleans, schema names, and
hashes of complete ordered case-ID lists. They serialize no individual case ID,
objective, patient fact, examination finding, test result, diagnosis, or other
source value.

## Required mechanics successor

A source pass authorizes only a separately frozen six-case mechanics gate. That
gate must establish before endpoint access:

- native patient and test responses add information beyond the fixed intake;
- strict atomic action grammar and missing-information behavior;
- no diagnosis or answer-key leakage into patient/test target prompts;
- at least four open generated diagnoses with nontrivial truth coverage;
- answer-conditioned support regeneration changes truth coverage;
- calibrated semantic response likelihoods and repeated-response stability;
- a source- or oracle-linked depth-two root that differs from compute-matched
  myopic and has positive first-link value;
- complete raw-response banking before gold diagnosis access.

Any source or mechanics failure closes this exact MedQA construction. There is
no row replacement, subset repair, threshold relaxation, or post-outcome prompt
repair.

## Accounting

- OpenRouter calls: `0`
- OpenRouter cost: `$0`
- OATML cluster use: none
