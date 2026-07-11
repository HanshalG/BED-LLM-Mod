# Paprika Best-N EIG Probe

The preregistered development-only best-N probe is endpoint-valid on tasks 0--9.
It changes only plain EIG candidate elicitation: request the five best distinct next
actions for resolving the issue quickly, then select by ordinary one-step EIG argmax.

| Arm | Resolution@5 | Mean censored turns |
|---|---:|---:|
| Best-N EIG | 0.50 | 4.0 |
| Naive-primary arbitration | 0.60 | 3.7 |
| Thinking naive | 0.40 | 4.9 |

Best-N EIG versus thinking naive has 4 wins, 1 loss, and 5 ties, with a mean paired
censored-turn difference of -0.9 and bootstrap 95% interval [-2.2, 0.4]. Versus
arbitration it has 2 wins, 4 losses, and 4 ties, with a +0.3 turn difference and interval
[-1.2, 1.6]. The probe is therefore directionally better than thinking naive and
statistically indistinguishable from arbitration at this sample size.

Under the decision rule written before launch, goal-anchored generate-and-select is a
competitive method identity. Arbitration remains the calibrated robustness variant,
but the held-out execution must add best-N EIG before headline outcomes are analyzed.
This probe does not enter the held-out estimate.
