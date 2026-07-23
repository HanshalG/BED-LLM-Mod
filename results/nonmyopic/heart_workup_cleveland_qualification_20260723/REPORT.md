# Cleveland Heart Workup Depth Qualification

Gate passed: **True**.

| Endpoint (d2 - d1) | Mean gain | Paired 95% CI | W/T/L |
| --- | ---: | ---: | ---: |
| Entropy AUC | +0.068747 | [+0.057311, +0.080303] | 171/87/39 |
| Truth-log AUC | +0.068747 | [+0.047312, +0.090220] | 153/87/57 |
| Final entropy | +0.030001 | [+0.016066, +0.048131] | 13/284/0 |
| Earlier workup rounds | +2.222222 | [+2.020202, +2.420875] | 210/87/0 |

## Mean Entropy Trace

| Round | d1 | d2 |
| ---: | ---: | ---: |
| 1 | 0.553454 | 0.553454 |
| 2 | 0.496062 | 0.509813 |
| 3 | 0.420170 | 0.405704 |
| 4 | 0.354539 | 0.263738 |
| 5 | 0.317591 | 0.174226 |
| 6 | 0.268535 | 0.048145 |
| 7 | 0.071136 | 0.006429 |
| 8 | 0.030001 | 0.000000 |

All planning, controls, posteriors, and metrics were exact and made zero LLM calls.
