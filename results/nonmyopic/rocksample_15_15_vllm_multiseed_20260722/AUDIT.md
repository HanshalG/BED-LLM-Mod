# RockSample[15,15] Direct-vLLM Multi-Seed Robustness

The preregistered two-fresh-seed robustness gate passes.

| Seed | Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | W/T/L |
| ---: | --- | --- | --- | --- |
| 24101 | Shared-roots d1 | +0.6723 [+0.6328, +0.7124] | +0.6088 [+0.4578, +0.7488] | 30/0/0 |
| 24101 | Exhaustive d1 width | +0.6700 [+0.6301, +0.7111] | +0.5837 [+0.4263, +0.7249] | 30/0/0 |
| 24101 | Random strategies | +0.6345 [+0.5722, +0.6877] | +0.5448 [+0.4139, +0.6640] | 30/0/0 |
| 24102 | Shared-roots d1 | +0.6619 [+0.6302, +0.6937] | +0.6947 [+0.5499, +0.8346] | 30/0/0 |
| 24102 | Exhaustive d1 width | +0.6590 [+0.6256, +0.6917] | +0.6932 [+0.5508, +0.8339] | 30/0/0 |
| 24102 | Random strategies | +0.6857 [+0.6454, +0.7255] | +0.6684 [+0.5462, +0.7798] | 30/0/0 |
| 24103 | Shared-roots d1 | +0.7176 [+0.6816, +0.7520] | +0.6750 [+0.4647, +0.8496] | 30/0/0 |
| 24103 | Exhaustive d1 width | +0.7157 [+0.6790, +0.7512] | +0.6672 [+0.4468, +0.8404] | 30/0/0 |
| 24103 | Random strategies | +0.7076 [+0.6714, +0.7431] | +0.6705 [+0.4649, +0.8357] | 30/0/0 |

## Pooled Three-Seed Estimates

| Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | W/T/L |
| --- | --- | --- | --- |
| Shared-roots d1 | +0.6839 [+0.6626, +0.7051] | +0.6595 [+0.5607, +0.7486] | 90/0/0 |
| Exhaustive d1 width | +0.6815 [+0.6603, +0.7036] | +0.6480 [+0.5479, +0.7369] | 90/0/0 |
| Random strategies | +0.6760 [+0.6493, +0.7017] | +0.6279 [+0.5373, +0.7082] | 90/0/0 |

The three runs made 3,964 requests at $0.00000000 API cost. Pooled intervals are secondary; the robustness claim requires both fresh seeds to pass independently.