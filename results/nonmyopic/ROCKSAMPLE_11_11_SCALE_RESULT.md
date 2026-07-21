# RockSample[11,11] Exact Scale Result

The preregistered zero-LLM structural gate passed. Positive gains favor exhaustive receding-horizon d2 over exhaustive d1.

| Endpoint | Paired gain | 95% CI | W/T/L |
| --- | ---: | --- | --- |
| Entropy AUC | +1.1042 | [+1.1015, +1.1068] | 500/0/0 |
| Truth-log AUC | +1.1129 | [+1.0885, +1.1377] | - |
| Final entropy | +2.0471 | [+2.0409, +2.0531] | - |

## Mechanism

Greedy d1 moves on `0/6000` decisions. Exact d2 moves on `4000/6000` decisions and uses `1` action sequence across all 500 truths:

`move-NORTH -> move-NORTH -> check-0 -> move-EAST -> move-EAST -> move-EAST -> check-4 -> move-EAST -> check-6 -> move-WEST -> move-SOUTH -> check-3`

Initial exact values are `{'1': 0.009185981503184948, '2': 0.0692831164199994}`. The independent audit reconstructed every paired AUC value from the raw traces. This result authorizes a separately preregistered root-slot serving smoke; it is not yet an LLM-policy result.
