# RockSample[15,15] Gemma 4 12B direct vLLM seed 24108

The registered fifteen-rock run passes its primary and truth-log corroboration gates. Positive paired gains favor StrategyEIG.

| Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |
| --- | --- | --- | --- |
| Shared-roots d1 | +0.6482 [+0.6030, +0.6937] | +0.6113 [+0.5270, +0.6918] | 30/0/0 |
| Exhaustive d1 width | +0.6102 [+0.5609, +0.6598] | +0.5378 [+0.4459, +0.6277] | 30/0/0 |
| Random strategies | +0.6277 [+0.5832, +0.6741] | +0.6505 [+0.5604, +0.7404] | 30/0/0 |

Primary gate passed: True. Truth-log corroboration passed: True. StrategyEIG moves on 183/450 decisions and captures 53.3% of exhaustive d2 value over nonterminal horizon-two rounds.

Its remaining entropy-AUC gap to exhaustive d2 is -0.4326 [-0.4853, -0.3797]. The run made 1290 physical requests, retained 0 rejected response, and cost $0.00000000. It had zero terminal failures, reasoning tokens, forced exits, or rollout-scoring LLM calls. The run did not resume from a prior failure.
