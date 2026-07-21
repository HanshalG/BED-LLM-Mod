# RockSample[11,11] Follow-Up Closure Result

The preregistered zero-call screen fails its paid-policy gate but cleanly localizes
the eleven-rock proposal gap.

| Root source | States | Original fraction | Closed fraction | Gain | Optimal-root coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma StrategyEIG | 330 | 0.7390 | 0.9686 | +0.2296 | 0.9606 |
| Matched random | 330 | 0.2304 | 0.9669 | +0.7366 | 0.9333 |

Gemma's roots are sufficient to recover nearly all exhaustive depth-two value once
their second actions are closed exactly, and 317/330 states include an optimal root.
However, random roots become almost equally strong after closure: their closed
fraction trails Gemma by only `0.0017`. This means the LLM prior's observed advantage
over matched random policies comes mainly from its continuation choices, not exclusive
access to valuable roots. The remaining Gemma closure loss is concentrated in rounds
10 and 11, whose mean closed fractions are `0.8344` and `0.8198`.

Two frozen conditions fail: Gemma's `0.9686` closed fraction is just below `0.97`,
and its closed advantage over random is below `0.03`. No paid closed-root run follows.
The analysis replays exact banked beliefs and makes no LLM calls, random draws, or
counterfactual policy claims.
