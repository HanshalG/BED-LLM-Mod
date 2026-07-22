# RockSample[15,15] Gemma 4 12B direct vLLM seed 24106

The registered fifteen-rock run passes its primary and truth-log corroboration gates. Positive paired gains favor StrategyEIG.

| Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |
| --- | --- | --- | --- |
| Shared-roots d1 | +0.6186 [+0.5843, +0.6565] | +0.6026 [+0.5393, +0.6652] | 30/0/0 |
| Exhaustive d1 width | +0.5792 [+0.5411, +0.6202] | +0.5560 [+0.4776, +0.6418] | 30/0/0 |
| Random strategies | +0.5946 [+0.5628, +0.6276] | +0.6422 [+0.5517, +0.7356] | 30/0/0 |

Primary gate passed: True. Truth-log corroboration passed: True. StrategyEIG moves on 173/450 decisions and captures 49.5% of exhaustive d2 value over nonterminal horizon-two rounds.

Its remaining entropy-AUC gap to exhaustive d2 is -0.4666 [-0.5028, -0.4284]. The run made 1357 physical requests, retained 0 rejected response, and cost $0.00000000. It had zero terminal failures, reasoning tokens, forced exits, or rollout-scoring LLM calls. It resumed from 372 revalidated cells after the reported prior error: cannot update on an impossible Rock Diagnosis observation.
