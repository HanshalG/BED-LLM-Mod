# Bongard Call-Matched Myopic Gate Sensitivity Audit

Date: 2026-08-09. Model calls: `0`. Endpoint data accessed: `false`.

This audit was frozen before the first Bongard response. It is a marginal
design-sensitivity calculation, not joint power and not an outcome forecast.
It does not alter a gate or authorize a paid call.

| Stage | N | True action-change rate for 80% power | Rank/log standardized margin | 80% Brier gain at SD 20% | 80% Brier gain at SD 30% |
|---|---:|---:|---:|---:|---:|
| development64_candidate | 64 | 41.93% | 0.105 SD | 5.10% | 6.31% |
| confirmation96_candidate | 96 | 41.22% | 0.086 SD | 5.72% | 8.58% |

The 37.5% changed-action floor has only about 50% marginal power when
the true rate sits at the floor. The Brier gate likewise needs a true
gain above 3% at realistic paired variance. Because dependencies among
Brier, action-change, ranking, and log-loss gates are unknown, full-tier
power cannot be inferred by multiplying these marginals and is bounded
above by the weakest one. A null must not trigger threshold relaxation.
