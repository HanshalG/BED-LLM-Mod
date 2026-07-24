# Detective CA-BED Aligned-Likelihood Diagnostic Result

Date: 2026-07-24

## Verdict

**Alignment repairs truth ranking but removes the apparent non-myopic
opportunity. Close Detective Cases.**

The exact hidden-role response function makes depth scores highly faithful on
the few cases with variation, but only 9.5% of questions distinguish a
murderer-role target from an innocent-role target. Most likelihood rows are
constant, so only 4/12 cases are rankable and depth two changes only 2/12 roots.

## Integrity

- Frozen source gate hash reproduced:
  `561e80787e88bf19c8c340a62fe61436aeccea91199616ab958fef65bbb9aca9`.
- Reused all 241 unique questions from the inspected 12-case tree.
- Exactly two fresh role calls per question: target murderer and target
  innocent.
- Exact `482/482` requests, zero reasoning tokens, zero forced exits, no
  response reuse, retry, parser repair, new question, or fresh case.
- Cost `$0.05618135`; `601,467` prompt and `3,176` completion tokens.

## Gates

| Metric | Required | Observed | Pass |
|---|---:|---:|:---:|
| Role-distinct questions | >=25% | **23/241 (9.5%)** | no |
| Rankable cases | >=9 | **4** | no |
| Different d2 root | >=4 | **2** | no |
| d2 truth-gain Spearman | >=.20 | **.933** | yes |
| d2 Spearman advantage | >=.10 | **-.033** | no |
| Mean d2 truth gain over d1 | >0 | **+.099** | yes |
| Strict d2 wins | >=6 | **2** | no |
| Mean d2 truth gain over random | >0 | **+.116** | yes |

## Interpretation

The original failure was genuine likelihood misspecification, not an arithmetic
bug. Replacing direct numerical introspection with the deployed semantic
response function changes d2 truth-rank correlation from `-.208` to `+.933`.

But the released Detective task does not define rich counterfactual worlds. A
targeted suspect receives the same private story whether any *other* suspect is
assumed to be the murderer. Under a coherent response model, the target has
only two roles: murderer or innocent. In 90.5% of generated questions Gemma
answers identically under both roles, giving a constant likelihood row and zero
EIG. The direct probability estimator created richer differences by inventing
counterfactual behavior, but those differences were not predictive of the
actual hidden-role answerer.

No prompt, smoothing, threshold, model, or split tuning follows. The 24 reserved
cases remain unused. Detective contributes a strong negative mechanism result,
not the desired LLM-native non-myopic positive.
