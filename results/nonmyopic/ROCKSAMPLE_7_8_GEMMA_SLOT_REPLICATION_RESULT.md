# RockSample[7,8] Root-Slot Replication

Gemma 4 26B A4B passes the preregistered root-slot replication under the same proposal interface used by GPT-5.4 Mini. Positive paired gains favor StrategyEIG.

| Model and interface | Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |
| --- | --- | --- | --- | --- |
| Gemma root slots | Shared-roots d1 | +0.7846 [+0.7370, +0.8296] | +0.7603 [+0.6769, +0.8409] | 30/0/0 |
| Gemma root slots | Exhaustive d1 width | +0.7729 [+0.7252, +0.8167] | +0.7183 [+0.6426, +0.7959] | 30/0/0 |
| Gemma root slots | Random strategies | +0.7334 [+0.6813, +0.7835] | +0.7954 [+0.6984, +0.8973] | 30/0/0 |
| GPT-5.4 Mini root slots | Shared-roots d1 | +0.5104 [+0.4312, +0.5863] | +0.4856 [+0.3746, +0.5956] | 28/0/2 |
| GPT-5.4 Mini root slots | Exhaustive d1 width | +0.4928 [+0.4115, +0.5734] | +0.4659 [+0.3503, +0.5805] | 28/0/2 |
| GPT-5.4 Mini root slots | Random strategies | +0.4130 [+0.3225, +0.4988] | +0.3987 [+0.2616, +0.5341] | 28/0/2 |

All six same-interface primary intervals and all six truth-log corroboration intervals exclude zero. The model-family difference is descriptive because the runs use different fresh seeds.

An independent auditor reconstructed every per-trial entropy-AUC and truth-log-AUC
contrast from the raw ten-step traces and matched the stored paired values, means,
and win/tie/loss counts to numerical tolerance.

## Interface Comparison

| Control | Gemma semantic gain | Gemma slot gain | Descriptive change |
| --- | ---: | ---: | ---: |
| Shared-roots d1 | +0.7502 | +0.7846 | +0.0344 |
| Exhaustive d1 width | +0.7297 | +0.7729 | +0.0432 |
| Random strategies | +0.6735 | +0.7334 | +0.0599 |

The Gemma slot run narrows the entropy-AUC gap to exhaustive d2 from -0.1303 to -0.0814, a descriptive improvement of +0.0490. Because the prompt runs use different seeds, this is robustness and mechanism evidence rather than a causal interface effect.

## Mechanics and Serving

| Run | Movement | h2 exhaustive fraction | Requests | Rejects | Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma semantic roots | 193/300 (0.643) | 0.858 | 904 | 34 | $0.21575537 |
| Gemma root slots | 199/300 (0.663) | 0.856 | 870 | 0 | $0.22487819 |
| GPT-5.4 Mini root slots | 191/300 (0.637) | 0.663 | 893 | 24 | $1.76258805 |

The fresh Gemma slot run completed all 870 cells without a rejection or resume. Every selected action was legal and exact rollout scoring made zero LLM calls.
