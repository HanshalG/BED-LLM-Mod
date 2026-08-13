# MedChain dynamic-support source protocol

Date frozen: 2026-08-13

Status: **value-blind source audit only; zero model calls**

## Scientific target

Use MedChain's released clinical cases as latent semantic worlds for adaptive
diagnosis. The policy sees only the chief complaint at intake. A separately
served standardized patient is grounded in the private current and past
history, physical examination, and auxiliary examination fields and reveals
them only in response to relevant questions. This directly avoids the closed
AgentClinic NEJM V1 interface, where the policy intake and patient grounding
were identical.

The intended first-link comparison, if every prerequisite passes, is depth-two
planning over answer-conditioned regenerated diagnostic support versus a
compute-matched myopic ensemble, fixed-support, history-blind, and random
controls. Source admission alone authorizes no model call or efficacy claim.

## Immutable source

- repository: `https://github.com/ljwztc/MedChain`
- commit: `35ab53e3c601831a7bca181c72ecad6b12b55ff3`
- tree: `ba425765442c69b51142ddd8d2731aa894eaf9b8`
- license: MIT
- data: `datasets/filtered_data_test_set.json`
- data SHA-256:
  `cb947c9b7ff8979ebbaa816b3b0977ed04bc84b6257147166a7fe725b7dbdb87`
- expected case population: `12163`
- patient interface: `doctor_patient_interaction/wenzhen_main.py`
- patient-interface SHA-256:
  `781aa0d9014c671f5fada3603d6292975e06b786b12d12d0f6df55a971677b08`
- workflow: `main.py`
- workflow SHA-256:
  `ac85f53576e1c871b72291e55941e07c2df275eabbe115433fc152cd6b776954`
- extraction helper: `utils/funtion_api.py`
- extraction-helper SHA-256:
  `ffa40c1899960c8b1cdd6ac022446280f84791badf0d0774e43593683bfeb3e0`

## Value-blind gates

The source passes only if every gate holds:

1. Repository, commit, tree, and bound file hashes match exactly.
2. The data is one JSON object containing exactly 12,163 unique nonempty case
   keys mapped to objects.
3. At least 90% of the full population and at least 1,000 cases satisfy the
   released patient-interface structure: a nonempty chief complaint; at least
   one nonempty current- or past-history field; and nonempty physical and
   auxiliary examination fields.
4. Every structurally eligible case has at least one nonempty diagnosis label
   in the released tags, and the eligible population contains at least 100
   distinct normalized diagnosis labels.
5. For every eligible case, the normalized policy-visible chief complaint is
   not canonically identical to any private history, physical-examination,
   auxiliary-examination, or diagnosis field.
6. The released patient code gives the doctor only the chief complaint, grounds
   the patient in chief complaint plus private history/examination fields, and
   explicitly instructs the patient not to volunteer examinations before a
   relevant question and not to invent missing facts.
7. The released workflow independently extracts chief complaint, histories,
   examinations, department, and diagnosis, and uses diagnosis only in the
   downstream diagnostic/evaluation path rather than the doctor intake.
8. The deterministic eligible-population split is complete and disjoint.

Case identity is SHA-256 of canonical compact JSON over the source key and row.
Eligible cases are ordered by
`SHA256("medchain-dynamic-support-v1|" + case_id)` and allocated:

| Split | Count | Access after source audit |
|---|---:|---|
| mechanics | 6 | only after a passing source audit and pushed protocol |
| opportunity | 30 | sealed |
| development | 64 | sealed |
| confirmation | 96 | sealed |
| reserve | remainder | sealed |

Public artifacts serialize only aggregate counts, booleans, schema-path names,
and hashes of complete ordered case-ID lists. They serialize no case key, case
ID, complaint, history, examination, image, department, diagnosis, treatment,
or other source value.

## Required mechanics successor

A source pass authorizes only a separately frozen six-case mechanics gate. It
must establish before any diagnosis endpoint opens:

- a strict atomic question grammar with one semantic fact request per action;
- patient answer obedience, missing-information behavior, and repeated-answer
  stability under exact prompts and seeds;
- no diagnosis or answer-key leakage into patient prompts;
- at least four generated diagnostic hypotheses with nontrivial truth coverage;
- answer-conditioned support regeneration that changes truth coverage;
- calibrated semantic response likelihoods under positive and negative answers;
- at least two dependent information actions before diagnosis;
- a depth-two root that differs from the compute-matched myopic ensemble and
  has positive oracle-linked first-link value on at least four of six cases;
- paired common-random-number diagnosis endpoints and a random control;
- complete candidate-score banking before gold diagnoses are opened.

Any source or mechanics failure closes this exact construction. There is no
case replacement, subset repair, threshold relaxation, endpoint-informed prompt
repair, or reuse of the failed AgentClinic interface.

## Accounting

- OpenRouter calls: `0`
- OpenRouter cost: `$0`
- OATML cluster use: none
