# ICAE-Bench Exact-10 Semantic Serving Result

Date: 2026-07-29

Status: **gated null; the exact serving interface is closed**.

## Frozen Run

- preregistration commit: `a700743`
- run: `icae-semantic-serving-20260729T052217Z`
- public serving SHA-256:
  `2e248dbf0a21bf5497c7ae18562ce6f478bb2c510187032f684445d628fd2f44`
- private raw SHA-256:
  `c6f3ed093f47e91910b36eff1bb4b48f5fa1f2253ed11a9933d641dc54a78748`
- selected mechanics tasks: `realcode@044` (JavaScript) and
  `realcode@276` (PHP)
- planner: `openai/gpt-5.4`, non-thinking, seed `50200`
- Oracle: `google/gemini-3.1-flash-lite`, non-thinking, seed `50300`

## Accounting

| Metric | Result |
|---|---:|
| accepted requests / expected | 10 / 10 |
| HTTP attempts / expected | 10 / 10 |
| retries / provider-error retries | 0 / 0 |
| reasoning tokens / forced exits | 0 / 0 |
| prompt / completion tokens | 92,610 / 3,864 |
| cost | `$0.0737541` |

All four planner supports parsed with exactly 12 unique hypotheses and six
unique questions. No response was repaired, normalized, reissued, or
substituted. No executable endpoint, development, confirmation, or retained
task was opened.

## Formal Failure

Two preregistered scientific gates failed:

| Gate | JavaScript | PHP | Required |
|---|---:|---:|---:|
| first question triggers a hidden requirement | no | yes | both |
| fresh-session Oracle reply is exact | yes | no | both |

The JavaScript first question was task-specific and plausible, but the strict
Oracle matched no released trigger phrase and returned the exact task
fallback. The PHP question matched one released test-contract entry in both
fresh sessions, yet the natural-language replies differed substantially.

The remaining semantic gates passed:

- the unrelated generic question produced the exact fallback with zero
  triggers on both tasks;
- all six followup questions changed after each realized reply; and
- refreshed supports reused four and one answer-introduced content terms,
  respectively.

## Saved-Response Diagnostic

This diagnostic uses only the already-paid responses and does not alter the
formal result. The PHP sessions agreed on the same trigger ID and identical
internal trigger log, but their replies had sequence similarity `0.625`.
Neither reply reproduced the injected fixed response for that trigger;
similarities were `0.027` and `0.028`.

The failure is therefore not transport noise or cosmetic formatting. The
released model-mediated Oracle can both miss a semantically relevant query
and paraphrase away from its own fixed response despite a stable trigger
classification.

## Interpretation

ICAE-Bench remains a strong source substrate: unmatched questions reveal
nothing, answers can change generated followup support, and independent
executable tests exist. This exact LLM factorization is not reliable enough
for Bayesian likelihoods or paired policy evaluation. A planner cannot assign
meaningful branch values when the same hidden record and question do not yield
a stable observation.

Do not rerun these seeds, change the first-question rule, loosen trigger
matching, normalize Oracle replies, replace the Oracle model, or proceed to a
mechanics policy test. A future ICAE route would require a separately
preregistered deterministic controller over the released trigger map, but that
would shift semantic matching away from the official LLM Oracle and must be
treated as a different method rather than a repair.
