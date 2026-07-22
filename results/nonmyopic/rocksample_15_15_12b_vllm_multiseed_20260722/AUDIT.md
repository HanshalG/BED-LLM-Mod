# RockSample[15,15] Gemma 4 12B Multi-Seed Robustness

The preregistered two-fresh-seed robustness gate passes.

| Seed | Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | W/T/L |
| ---: | --- | --- | --- | --- |
| 24106 | Shared-roots d1 | +0.6186 [+0.5843, +0.6565] | +0.6026 [+0.5393, +0.6652] | 30/0/0 |
| 24106 | Exhaustive d1 width | +0.5792 [+0.5411, +0.6202] | +0.5560 [+0.4776, +0.6418] | 30/0/0 |
| 24106 | Random strategies | +0.5946 [+0.5628, +0.6276] | +0.6422 [+0.5517, +0.7356] | 30/0/0 |
| 24107 | Shared-roots d1 | +0.6584 [+0.6112, +0.7064] | +0.6940 [+0.6297, +0.7556] | 30/0/0 |
| 24107 | Exhaustive d1 width | +0.6330 [+0.5870, +0.6786] | +0.6862 [+0.5920, +0.7840] | 30/0/0 |
| 24107 | Random strategies | +0.6359 [+0.5791, +0.6911] | +0.5827 [+0.4820, +0.6779] | 30/0/0 |
| 24108 | Shared-roots d1 | +0.6482 [+0.6030, +0.6937] | +0.6113 [+0.5270, +0.6918] | 30/0/0 |
| 24108 | Exhaustive d1 width | +0.6102 [+0.5609, +0.6598] | +0.5378 [+0.4459, +0.6277] | 30/0/0 |
| 24108 | Random strategies | +0.6277 [+0.5832, +0.6741] | +0.6505 [+0.5604, +0.7404] | 30/0/0 |

## Pooled Three-Seed Estimates

| Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | W/T/L |
| --- | --- | --- | --- |
| Shared-roots d1 | +0.6417 [+0.6171, +0.6672] | +0.6360 [+0.5960, +0.6752] | 90/0/0 |
| Exhaustive d1 width | +0.6075 [+0.5813, +0.6340] | +0.5933 [+0.5422, +0.6446] | 90/0/0 |
| Random strategies | +0.6194 [+0.5930, +0.6451] | +0.6251 [+0.5718, +0.6773] | 90/0/0 |

The three runs made 3,937 requests at $0.00000000 API cost. Pooled intervals are secondary; the robustness claim requires both fresh seeds to pass independently.