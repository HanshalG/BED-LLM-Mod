# Mushroom Feature Acquisition Depth Qualification

Gate passed: **True**.

| Endpoint (d2 - d1) | Mean gain | Paired 95% CI | W/T/L |
| --- | ---: | ---: | ---: |
| Entropy AUC | +0.115348 | [+0.105676, +0.125362] | 693/0/307 |
| Truth-log AUC | +0.117191 | [+0.102234, +0.132444] | 542/0/458 |
| Final entropy | +0.000000 | [+0.000000, +0.000000] | 0/1000/0 |

## Mean Entropy Trace

| Round | d1 | d2 |
| ---: | ---: | ---: |
| 1 | 0.556368 | 0.692501 |
| 2 | 0.369863 | 0.059965 |
| 3 | 0.258915 | 0.017898 |
| 4 | 0.200257 | 0.004875 |
| 5 | 0.183686 | 0.000000 |
| 6 | 0.128931 | 0.000000 |
| 7 | 0.000000 | 0.000000 |
| 8 | 0.000000 | 0.000000 |

All planning, posterior updates, controls, and metrics were exact and made zero LLM calls.
