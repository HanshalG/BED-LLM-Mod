# ClinDiag Deterministic De-Anchored Refresh Result

Date: 2026-07-24

Status: **failed stability; de-anchored ClinDiag refresh is closed.**

## Result

The sole temperature-zero calibration completed exactly 10 requests with zero
reasoning/retries. All eight supports parsed to size 12, duplicate prompts were exact,
source evidence contained no full target, and duplicate truth-score gaps were zero.

Exact-prompt semantic overlap still failed:

| Case | Truth-score gap | Worse directional overlap |
|---|---:|---:|
| `23697517` | 0.00 | 0.1667 |
| `rare216` | 0.00 | 0.5000 |

Both truths were already present and remained present, so no efficacy endpoint exists.

## Interpretation

Temperature was not the cause of de-anchored set instability. At temperature zero,
GPT-5.4 retained the correct disease but exchanged many clinically related
alternatives:

- the Addison case varied among nutritional, endocrine, autoimmune, and autonomic
  mimics;
- the familial-HLH case varied among EBV-associated lymphoproliferative disorders and
  immune deficiencies.

Manual inspection finds more related cross-list overlap than the strict evaluator
credited, but still below the frozen same-disease/synonym `0.80` rule. The threshold
cannot be changed after observing the result.

## Decision

The de-anchored prompt family is closed. No further temperature, overlap, prompt, or
replay tuning follows, and no headroom, pair, likelihood, scorer, or policy run is
authorized.

Together with the prior-retaining pair null, ClinDiag establishes a useful tradeoff:
retaining the previous support is reproducible but too inert to unlock truth in two
steps; rebuilding from evidence is more responsive but not reproducible enough at the
12-diagnosis set level. The project should move to a different belief representation
or environment rather than tune this interface further.

## Cost

- exactly 10 requests and zero reasoning;
- 4,596 prompt and 1,402 completion tokens;
- `$0.02516650`;
- conservative project-ledger remainder: `$21.82849872`.
