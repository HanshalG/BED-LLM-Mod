# RockSample[15,15] GPT-5.4 Mini OpenRouter seed 24114

The registered fifteen-rock run passes its primary and truth-log corroboration gates. Positive paired gains favor StrategyEIG.

| Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |
| --- | --- | --- | --- |
| Shared-roots d1 | +0.4716 [+0.3762, +0.5601] | +0.5236 [+0.3965, +0.6481] | 28/0/2 |
| Exhaustive d1 width | +0.4562 [+0.3653, +0.5482] | +0.5553 [+0.4003, +0.7054] | 28/0/2 |
| Random strategies | +0.4410 [+0.3245, +0.5483] | +0.5276 [+0.3984, +0.6509] | 27/0/3 |

Primary gate passed: True. Truth-log corroboration passed: True. StrategyEIG moves on 233/450 decisions and captures 45.7% of exhaustive d2 value over nonterminal horizon-two rounds.

Its entropy-AUC difference from exhaustive d2 is -0.0714 [-0.1623, +0.0208]. The run made 1372 physical requests, retained 60 rejected responses, and cost $3.68507010. It had zero terminal failures, reasoning tokens, forced exits, or rollout-scoring LLM calls. The run did not resume from a prior failure.
