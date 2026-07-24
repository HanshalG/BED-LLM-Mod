# ClinDiag Multiple-Choice Joint-Model Gate Result

Date: 2026-07-24

Status: **serving smoke failed; no structural lookahead, planner, or holdout is
authorized for this exact interface.**

## Frozen Protocol

Four fresh ClinDiag cases received:

- one 12-diagnosis support generated from the initial presentation;
- four target-blind multiple-choice clinical queries with four outcomes each;
- one hidden-case gatekeeper response per query;
- one exact duplicate gatekeeper request per case;
- one likelihood matrix over six generated hypotheses plus the true diagnosis added
  only after queries and realized outcomes were frozen;
- one independent behavioral audit per case.

GPT-5.4 generated supports and queries. GPT-5.4 Mini served as gatekeeper, likelihood
model, and audit judge. Every role ran without reasoning and without retries.

## Result

| Criterion | Required | Observed | Pass |
|---|---:|---:|---:|
| Physical requests | 36 | 36 | yes |
| Reasoning tokens | 0 | 0 | yes |
| Parsed supports | 4 of size 12 | 4 of size 12 | yes |
| Valid query/option schemas | 16 | 16 | yes |
| Behavioral response audits | 16/16 | 10/16 | **no** |
| Exact duplicate outcome matches | 4/4 | 4/4 | yes |
| Duplicate semantic consistency | 4/4 | 4/4 | yes |
| Literal target leaks | 0 | 0 | yes |
| Mean `p(realized | true)` | at least .35 | .5663 | yes |
| Positive true-vs-generated margins | at least 8/16 | 10/16 | yes |
| Mean true-vs-generated margin | at least +.05 | +.1743 | yes |

The only failed gate was environment behavior, but it is load-bearing.

## Failure Diagnosis

Five responses failed case consistency and one response failed the frozen no-target-
leak audit.

Concrete environment errors included:

- an unrecorded TSH request returned “TSH not measured in the record,” labeled it
  recorded, and mapped it to the unavailable/assay-failure option instead of
  synthesizing a patient-consistent result;
- a synthetic punch biopsy for the Porokeratosis of Mibelli case selected localized
  epidermolytic hyperkeratosis, which the audit judged inconsistent with the hidden
  case;
- several synthetic cardiac, ventilation-perfusion, and toxicology findings were
  judged insufficiently supported by the hidden record.

The no-leak failure concerned a recorded high-resolution chest CT response in the
mitral-leaflet case. That individual judgment is debatable because the response did
not literally name the target; nevertheless, the audit and threshold were frozen
before serving, so it remains a failure rather than being post-hoc excluded.

The candidate generator also proposed invasive confirmatory biopsies in two cases.
That did not violate this smoke's schema, but it would risk restoring the same
one-action saturation that invalidated the coarse native-block route.

## Positive Mechanism Signal

The enumerable likelihood representation behaved much better than the environment:

- all 28 likelihood rows per case parsed and normalized;
- mean realized-outcome probability under the true diagnosis was `.5663`;
- the true diagnosis exceeded the six generated hypotheses on 10/16 queries;
- the mean probability margin was `+.1743`;
- exact duplicate outcome IDs and semantic findings agreed on all four controls.

This supports retaining the BED-LLM-style multiple-choice outcome space and explicit
`p(y | history, hypothesis, query)`. It does not validate the GPT-5.4 Mini hidden
gatekeeper.

## Cost And Consequence

- 36 requests;
- 28,139 prompt tokens and 8,193 completion tokens;
- zero reasoning tokens, retries, or forced exits;
- cost: `$0.08538475`;
- conservative project-ledger balance after the run: `$23.93270872`.

The exact Mini-gatekeeper interface stops here. A future attempt must be
preregistered as a distinct environment-model qualification, use a stronger
non-reasoning gatekeeper, explicitly exclude invasive confirmatory actions, and
retain exact duplicates plus independent behavioral audits. It may reuse the
multiple-choice likelihood construction, but it cannot be described as a passed
repair of this smoke.
