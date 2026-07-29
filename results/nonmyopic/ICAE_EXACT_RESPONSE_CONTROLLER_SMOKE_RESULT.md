# ICAE Semantic-Matcher Exact-Response Controller Result

Date: 2026-07-29

Status: **all serving gates pass; a separately frozen mechanics first-link
experiment is authorized**.

## Frozen Run

- preregistration commit: `05990ca`
- run: `icae-exact-controller-20260729T052833Z`
- public serving SHA-256:
  `a5e7731b7ebcfb6d89b5634c660eae47dc74fc84c4540fee6225a685a12879ca`
- private raw SHA-256:
  `378692c4f840974b7e04275df031cf1aac8261988a9982536e93de715ebdaed3`
- selected tasks: `realcode@235` (Kotlin) and `realcode@185` (Go)

## Result

| Metric | Result |
|---|---:|
| accepted requests / expected | 10 / 10 |
| HTTP attempts / expected | 10 / 10 |
| retries / provider-error retries | 0 / 0 |
| reasoning tokens / forced exits | 0 / 0 |
| prompt / completion tokens | 18,675 / 3,021 |
| cost | `$0.05894365` |

Every preregistered gate passed:

- GPT-5.4 produced strict 12-hypothesis/six-question initial and refreshed
  supports on both tasks;
- GPT-5.4 Mini matched each first question to exactly one released trigger;
- repeated semantic matches were exact on both tasks;
- generic questions returned no trigger and the exact stored fallback;
- code returned the exact released response text without exposing that text to
  the matcher;
- all six followup questions changed on both tasks; and
- refreshed supports reused two and three answer-introduced content terms.

## Interpretation

This isolates the failure in the official-style smoke. The LLM can reliably
own free-form semantic routing when its output is a constrained trigger ID.
Instability came from asking the same model to generate the observation text
after classification. Returning the released response programmatically makes
the observation model deterministic without replacing semantic matching with
string search or exposing answers to the matcher.

This is serving qualification, not non-myopic efficacy. It authorizes only a
separately preregistered mechanics first-link experiment with paired,
equal-query policies. Development, confirmation, retained, and executable
coding endpoints remain unopened.
