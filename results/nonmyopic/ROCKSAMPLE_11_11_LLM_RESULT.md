# RockSample[11,11] Gemma Root-Slot Confirmation

The preregistered eleven-rock scale confirmation passes its primary and truth-log corroboration gates. Positive paired gains favor StrategyEIG.

| Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |
| --- | --- | --- | --- |
| Shared-roots d1 | +0.9412 [+0.8968, +0.9807] | +0.9258 [+0.8155, +1.0306] | 30/0/0 |
| Exhaustive d1 width | +0.9387 [+0.8952, +0.9773] | +0.9219 [+0.8129, +1.0278] | 30/0/0 |
| Random strategies | +0.8867 [+0.8305, +0.9385] | +0.9070 [+0.7750, +1.0428] | 30/0/0 |

All three primary and all three truth-log intervals exclude zero. StrategyEIG moves on 195/360 decisions and captures 73.9% of exhaustive d2 value over nonterminal horizon-two rounds.

The remaining entropy-AUC gap to exhaustive d2 is -0.1629 [-0.2009, -0.1295]. The run made 1051 physical requests, retained 1 rejected response, and cost $0.30078251. It had zero terminal failures, reasoning tokens, forced exits, resumes, or rollout-scoring LLM calls.
