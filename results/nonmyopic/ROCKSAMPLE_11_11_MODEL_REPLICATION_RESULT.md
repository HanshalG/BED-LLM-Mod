# RockSample[11,11] Cross-Model Root-Slot Replication

Both preregistered model runs pass all entropy-AUC and truth-log-AUC gates under the identical ordered root-slot interface. Positive gains favor StrategyEIG.

| Model | Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |
| --- | --- | --- | --- | --- |
| Gemma 4 26B A4B | Shared-roots d1 | +0.9412 [+0.8968, +0.9807] | +0.9258 [+0.8155, +1.0306] | 30/0/0 |
| Gemma 4 26B A4B | Exhaustive d1 width | +0.9387 [+0.8952, +0.9773] | +0.9219 [+0.8129, +1.0278] | 30/0/0 |
| Gemma 4 26B A4B | Random strategies | +0.8867 [+0.8305, +0.9385] | +0.9070 [+0.7750, +1.0428] | 30/0/0 |
| GPT-5.4 Mini | Shared-roots d1 | +0.5811 [+0.4672, +0.6909] | +0.6053 [+0.4244, +0.7750] | 29/0/1 |
| GPT-5.4 Mini | Exhaustive d1 width | +0.5528 [+0.4374, +0.6613] | +0.6066 [+0.4033, +0.7988] | 27/0/3 |
| GPT-5.4 Mini | Random strategies | +0.4688 [+0.3364, +0.5919] | +0.5183 [+0.3255, +0.7011] | 27/0/3 |

All twelve cross-model intervals exclude zero. Effect-size differences are descriptive because the models use different fresh policy seeds.

| Model | Movement | h2 exhaustive fraction | Exact-d2 AUC gap | Requests | Rejects | Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 26B A4B | 195/360 (0.542) | 0.739 | -0.1629 | 1051 | 1 | $0.30078251 |
| GPT-5.4 Mini | 231/360 (0.642) | 0.626 | -0.5459 | 1069 | 20 | $2.54133255 |

GPT's smaller gains, lower exhaustive fraction, and larger exact-d2 gap show model-dependent proposal quality, but not a model-dependent sign. Both runs use one proposal call per StrategyEIG decision and zero rollout-scoring LLM calls.
