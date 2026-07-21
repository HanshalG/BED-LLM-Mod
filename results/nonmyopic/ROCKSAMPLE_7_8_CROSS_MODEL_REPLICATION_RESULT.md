# RockSample[7,8] Cross-Model Replication

The preregistered positive non-myopic result replicates across Gemma 4 26B A4B and GPT-5.4 Mini. Positive paired gains favor StrategyEIG.

| Model | Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |
| --- | --- | --- | --- | --- |
| Gemma 4 26B A4B | Shared-roots d1 | +0.7502 [+0.7050, +0.7938] | +0.6622 [+0.6029, +0.7206] | 30/0/0 |
| Gemma 4 26B A4B | Exhaustive d1 width | +0.7297 [+0.6874, +0.7724] | +0.6511 [+0.5813, +0.7303] | 30/0/0 |
| Gemma 4 26B A4B | Random strategies | +0.6735 [+0.6251, +0.7212] | +0.5841 [+0.5158, +0.6554] | 30/0/0 |
| GPT-5.4 Mini | Shared-roots d1 | +0.5104 [+0.4312, +0.5863] | +0.4856 [+0.3746, +0.5956] | 28/0/2 |
| GPT-5.4 Mini | Exhaustive d1 width | +0.4928 [+0.4115, +0.5734] | +0.4659 [+0.3503, +0.5805] | 28/0/2 |
| GPT-5.4 Mini | Random strategies | +0.4130 [+0.3225, +0.4988] | +0.3987 [+0.2616, +0.5341] | 28/0/2 |

Both model families pass all three registered primary comparisons and all truth-log corroboration intervals. GPT-5.4 Mini's gains are smaller; this is a descriptive cross-seed difference, not a registered model-superiority test.

## Mechanism and Serving

| Model | StrategyEIG movement | h2 exhaustive fraction | Requests | Rejects | Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 26B A4B | 193/300 (0.643) | 0.858 | 904 | 34 | $0.21575537 |
| GPT-5.4 Mini | 191/300 (0.637) | 0.663 | 893 | 24 | $1.76258805 |

Every selected action was legal, initial strategy cells were shared with d1, the width control was compute matched, and exact rollout scoring made zero LLM calls.
