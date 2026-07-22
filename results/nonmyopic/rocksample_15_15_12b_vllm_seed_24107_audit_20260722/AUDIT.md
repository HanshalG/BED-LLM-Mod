# RockSample[15,15] Gemma 4 12B direct vLLM seed 24107

The registered fifteen-rock run passes its primary and truth-log corroboration gates. Positive paired gains favor StrategyEIG.

| Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |
| --- | --- | --- | --- |
| Shared-roots d1 | +0.6584 [+0.6112, +0.7064] | +0.6940 [+0.6297, +0.7556] | 30/0/0 |
| Exhaustive d1 width | +0.6330 [+0.5870, +0.6786] | +0.6862 [+0.5920, +0.7840] | 30/0/0 |
| Random strategies | +0.6359 [+0.5791, +0.6911] | +0.5827 [+0.4820, +0.6779] | 30/0/0 |

Primary gate passed: True. Truth-log corroboration passed: True. StrategyEIG moves on 195/450 decisions and captures 52.4% of exhaustive d2 value over nonterminal horizon-two rounds.

Its remaining entropy-AUC gap to exhaustive d2 is -0.4212 [-0.4712, -0.3712]. The run made 1290 physical requests, retained 0 rejected response, and cost $0.00000000. It had zero terminal failures, reasoning tokens, forced exits, or rollout-scoring LLM calls. The run did not resume from a prior failure.
