# RockSample[11,11] Proposal-Width Robustness

The preregistered all-width gate passes.

| K | Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | W/T/L |
| ---: | --- | --- | --- | --- |
| 2 | Shared-roots d1 | +0.3686 [+0.2921, +0.4381] | +0.3828 [+0.2704, +0.5005] | 26/0/4 |
| 2 | Exhaustive d1 width | +0.3559 [+0.2817, +0.4232] | +0.3466 [+0.2227, +0.4671] | 24/0/6 |
| 2 | Random strategies | +0.3728 [+0.2765, +0.4543] | +0.3560 [+0.2474, +0.4649] | 25/0/5 |
| 4 | Shared-roots d1 | +0.9231 [+0.9053, +0.9425] | +0.8994 [+0.7947, +1.0094] | 30/0/0 |
| 4 | Exhaustive d1 width | +0.9207 [+0.9013, +0.9427] | +0.9015 [+0.7993, +1.0045] | 30/0/0 |
| 4 | Random strategies | +0.9047 [+0.8574, +0.9453] | +0.9866 [+0.9015, +1.0706] | 30/0/0 |
| 6 | Shared-roots d1 | +0.9365 [+0.9160, +0.9574] | +0.8938 [+0.7976, +0.9978] | 30/0/0 |
| 6 | Exhaustive d1 width | +0.9310 [+0.9146, +0.9486] | +0.9000 [+0.8038, +1.0039] | 30/0/0 |
| 6 | Random strategies | +0.8453 [+0.7893, +0.8950] | +0.8306 [+0.7234, +0.9401] | 30/0/0 |

## Secondary Width Comparisons

Positive gains favor the larger K.

| Comparison | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] |
| --- | --- | --- |
| k4 - k2 | +0.5655 [+0.4974, +0.6418] | +0.5555 [+0.4794, +0.6363] |
| k6 - k4 | +0.0102 [-0.0156, +0.0337] | -0.0004 [-0.0433, +0.0440] |
| k6 - k2 | +0.5757 [+0.5026, +0.6550] | +0.5551 [+0.4768, +0.6426] |

The three runs made 3,157 requests and cost $0.79891053. Cross-width differences are secondary; the primary claim is positive gain at every K.
