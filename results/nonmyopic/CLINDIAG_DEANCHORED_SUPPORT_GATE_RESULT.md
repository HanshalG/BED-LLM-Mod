# ClinDiag De-Anchored Full-Refresh Serving Result

Date: 2026-07-24

Status: **failed stability; no efficacy screen at temperature 0.5.**

## Result

The run completed exactly 10 requests with zero reasoning/retries. All eight supports
parsed to size 12, duplicate prompts were exact, no full target string appeared in
source evidence, and both duplicate truth-score gaps were zero.

The semantic support-overlap gate failed identically on both cases:

| Case | Truth-score gap | Worse directional overlap |
|---|---:|---:|
| `21991897` | 0.00 | 0.50 |
| `rare70` | 0.00 | 0.50 |

Both truths were already present initially and remained present, so this smoke contains
no efficacy endpoint. The failure is exact-prompt set instability.

## Mechanism Audit

Removing the prior-inclusion instruction changed approximately half of each 12-item
terminal differential under exact replay. The lists remained clinically coherent and
retained the truth, but they varied among related renal-crisis mimics and febrile-rash
infections.

This confirms the intended mechanism change reduced anchoring, but at temperature
`0.5` it traded support inertia for too much list variance. The frozen `0.80` overlap
threshold cannot be relaxed after seeing the result.

## Decision

No headroom, pair, likelihood, scorer, or policy run follows from this serving result.
The exact de-anchored temperature-0.5 line is closed.

One distinct deterministic calibration may be preregistered on fresh cases with the
same prompt and generation temperature fixed to `0.0`. It must keep the same overlap
and truth-score thresholds; failure closes de-anchoring entirely.

## Cost

- exactly 10 requests and zero reasoning;
- 4,564 prompt and 1,328 completion tokens;
- `$0.02467650`;
- conservative project-ledger remainder: `$21.85366522`.
