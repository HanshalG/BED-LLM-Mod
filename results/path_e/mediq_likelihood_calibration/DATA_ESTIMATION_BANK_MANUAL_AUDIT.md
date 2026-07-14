# MediQ Data-Estimation Bank Manual Audit

Date: 2026-07-14

Status: **diagnostic audit after the automated bank was declared inadequate**.
This is not a formal calibration-bank pass. Under the frozen preregistration, the
bank needed at least 15 available Yes/No outcomes before likelihood scoring. It
contained 8, so the scorer was not run and Claim 1 remains blocked.

Run:
`20260714T051844_mediq-data-estimation-bank-naive-nonthinking-26b-seed1304-v3`

## Automated Result

- 10 held-out iMEDQA cases (source offsets 5-14), 3 turns each.
- 30 canonical, non-duplicate, semantically valid Yes/No questions.
- 30/30 replies mapped cleanly, were grounded in the official atomic-fact record,
  and passed the relevance contract.
- 8/30 replies were available Yes/No; 22/30 were unavailable.
- No scorer calls were made during bank generation.
- The only failed validation check was the preregistered minimum of 15 available
  outcomes.

## Transcript Audit

| Source | Exam target | Selected question | Outcome | Manual finding |
|---:|---|---|---|---|
| 5 | Etiology of anemia: iron deficiency | History of heavy menstrual bleeding? | Unavailable | Clinically useful and atomic; absent from record. |
| 5 | Same | History of pica? | Unavailable | Clinically useful and atomic; absent from record. |
| 5 | Same | History of gastrointestinal bleeding? | Unavailable | Clinically useful and atomic; absent from record. |
| 6 | Mechanism of temporal wasting: proteasomal degradation | Visible temporal-muscle wasting? | Yes | Directly supported by bilateral temporal wasting. |
| 6 | Same | Unintentional weight loss? | Yes | Directly supported by 10.5 kg loss in three months. |
| 6 | Same | Fever in the past three months? | Unavailable | Atomic and relevant; absent from record. |
| 7 | Diagnosis: non-exertional heat stroke | Recent antipsychotic medication? | Yes | Directly supported by administered haloperidol. |
| 7 | Same | Muscle rigidity present? | Unavailable | Important discriminator from NMS; absent from record. |
| 7 | Same | Temperature above 104 F? | No | Directly contradicted by recorded 102.9 F temperature. |
| 8 | Enzyme: alpha-ketoglutarate dehydrogenase | Extremity numbness? | Yes | Directly supported by hand and foot numbness. |
| 8 | Same | History of veganism? | Unavailable | Reasonable thiamine-risk question; absent from record. |
| 8 | Same | Leg weakness? | Yes | Directly supported by lower-extremity weakness. |
| 9 | Next management step: atenolol | Blood pressure above 140/90? | Yes | Directly supported by 147/98 mmHg. |
| 9 | Same | Diabetes mellitus? | Unavailable | Relevant comorbidity; absent from record. |
| 9 | Same | Chronic kidney disease? | Unavailable | Relevant comorbidity; absent from record. |
| 10 | Hypothalamic hormone effect: decreased growth hormone | Peptic-ulcer history? | Unavailable | Medically meaningful but weakly tied to the exam target; absent. |
| 10 | Same | Nausea? | Unavailable | Medically meaningful but weakly tied to the exam target; absent. |
| 10 | Same | Diabetes mellitus? | Unavailable | Medically meaningful but weakly tied to the exam target; absent. |
| 11 | Dysthymia duration: two years | Prior major depressive episode? | Unavailable | Relevant differential/criteria question; absent from record. |
| 11 | Same | Two-week major depressive period? | Unavailable | Relevant criteria question; absent from record. |
| 11 | Same | Symptoms continuously for at least two years? | Yes | Directly supported by symptoms for more days than not for three years. |
| 12 | Boundary management: closed questions and chaperone | Sexual advances toward physician? | Unavailable | The vignette implies attraction but does not explicitly record an advance. Strict FactSelect behavior is correct. |
| 12 | Same | Sexually suggestive physical gestures? | Unavailable | No explicit supporting fact. |
| 12 | Same | Verbal comments of a sexual nature? | Unavailable | No explicit supporting fact under the strict entailment contract. |
| 13 | Brown gallstone cause: E. coli beta-glucuronidase | Fever? | Unavailable | Relevant to infection; absent from record. |
| 13 | Same | Diarrhea? | Unavailable | Potentially relevant; absent from record. |
| 13 | Same | Jaundice? | Unavailable | Clinically useful and atomic; absent from record. |
| 14 | Cause of rhythm: monomorphic ventricular tachycardia | Family history of sudden cardiac death? | Unavailable | Relevant discriminator; absent from text record. |
| 14 | Same | History of syncope? | Unavailable | Relevant discriminator; absent from text record. |
| 14 | Same | Lightheadedness? | Unavailable | Relevant symptom; absent from text record. |

## Interpretation

The bank did not fail because the policy emitted malformed, duplicated, compound, or
unmedical questions. It failed because the official FactSelect patient can expose only
facts already present in a short static record. Most reasonable discriminating facts
are simply not annotated.

At the observed answerability rate `a = 8/30`, an unavailable-neutral binary question
has the loose information ceiling

```text
I(theta; response | query) <= a * log(2) = 0.1848 nats.
```

Three questions therefore have a loose additive ceiling of 0.5545 nats before also
applying the initial target-entropy ceiling and accounting for imperfect diagnostic
specificity. This leaves little signal for distinguishing a depth-2 policy from greedy
EIG, even with a perfect likelihood model.

The observed missingness also depends heavily on the source record, not merely on the
diagnosis. Treating unavailable as label-informative would exploit annotation density;
treating it as neutral, as required for a defensible patient model, exposes the sparse
information channel.

## Decision

Per `DATA_ESTIMATION_PREREGISTRATION.md`:

1. Do not run the frozen data-estimation scorer on this inadequate bank.
2. Do not tune the availability threshold, prompts, or selected cases against these
   outcomes.
3. Do not launch the MediQ Claim-1 or depth-2 study.
4. Treat the 8/30 result as evidence that iMEDQA's static-record channel is not a
   suitable non-myopic BED test under the official FactSelect contract.
