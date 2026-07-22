# RockSample[15,15] Gemma 4 E4B direct vLLM seed 24105

The registered fifteen-rock run fails its preregistered primary entropy-AUC gate.

| Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |
| --- | --- | --- | --- |
| Shared-roots d1 | +0.0687 [+0.0254, +0.1208] | +0.0753 [-0.0045, +0.1602] | 19/0/11 |
| Exhaustive d1 width | +0.0408 [-0.0035, +0.0884] | -0.0029 [-0.0871, +0.0814] | 17/0/13 |
| Random strategies | +0.0518 [+0.0010, +0.1098] | +0.1076 [+0.0285, +0.1939] | 18/0/12 |

Primary gate passed: False. Truth-log corroboration passed: False. StrategyEIG moves on 121/450 decisions and captures 6.8% of exhaustive d2 value over nonterminal horizon-two rounds.

Its remaining entropy-AUC gap to exhaustive d2 is -0.5517 [-0.5884, -0.5112]. The run made 1294 physical requests, retained 5 rejected response, and cost $0.00000000. It had zero terminal failures, reasoning tokens, forced exits, resumes, or rollout-scoring LLM calls.
