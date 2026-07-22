# Frozen RockSample[15,15] Exact Scale Result

The preregistered zero-LLM structural gate passed on 100 paired trials.

| Endpoint | d2 gain over d1 | Paired 95% CI |
| --- | ---: | --- |
| Entropy AUC | +0.5888 | [0.5800562532034176, 0.5970613129132408] |
| Truth-log AUC | +0.5930 | [0.547015567301299, 0.6396036301945324] |
| Final entropy | +1.3882 | [1.3654625474817803, 1.4100238791613497] |

Entropy-AUC wins/ties/losses were `[100, 0, 0]`. Greedy d1 moved on `0/1500` decisions; exact d2 moved on `500/1500`.

The independent audit reconstructed every paired value and bootstrap interval. Initial exact values were `{'1': 0.0057223690223843215, '2': 0.02985110651988665}`.
