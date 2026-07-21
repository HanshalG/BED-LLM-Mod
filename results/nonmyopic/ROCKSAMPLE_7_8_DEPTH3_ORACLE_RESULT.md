# RockSample[7,8] Exact Depth-Three Result

The preregistered incremental d3-over-d2 gate failed. Positive paired gains favor the deeper arm.

| Comparison | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | W/T/L |
| --- | --- | --- | --- |
| d2 minus d1 | +0.8509 [+0.8468, +0.8548] | +0.8499 [+0.8278, +0.8726] | 500/0/0 |
| d3 minus d2 | -0.2044 [-0.2090, -0.1998] | -0.2196 [-0.2396, -0.1999] | 0/0/500 |
| d3 minus d1 | +0.6465 [+0.6404, +0.6526] | +0.6304 [+0.6015, +0.6588] | 500/0/0 |

## Mechanism

D2 and d3 take the same first five actions, then diverge at round 6 from position `(2, 5)`. D2 values `move-EAST` at `0.6931` nats, ahead of immediate `check-6` at `0.1386`. D3 instead values `check-6` at `0.7624`, ahead of `move-EAST` at `0.7230`: check now, then move and check perfectly.

After the noisy check, constant-horizon replanning restores a three-step window and makes the same promise again. D3 repeats `check-6` for three rounds instead of executing the promised move. D2 moves immediately and obtains a perfect check. This is a deterministic receding-horizon commitment failure: all 500 d2 trajectories and all 500 d3 trajectories use their respective fixed action sequences.

- Primary gate: **False**.
- Independent trace audit: **True**.
- Prior exact-d2 result reproduced: **True**.
- Consequence: the frozen gate stops paid h3 LLM-policy engineering.
