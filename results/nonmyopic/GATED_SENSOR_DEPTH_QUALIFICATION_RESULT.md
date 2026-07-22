# Gated Sensor Exact Depth Qualification

## Result

The preregistered structural gate passed on 500 paired trials. Exact depth-two planning exploited zero-information setup actions and substantially outperformed exact greedy design over eight rounds.

| Endpoint | d2 gain over d1 | Paired 95% CI |
| --- | ---: | --- |
| Entropy AUC | `+1.2459` nats | `[+1.2401, +1.2516]` |
| Truth-log AUC | `+1.2806` | `[+1.1943, +1.3630]` |
| Final entropy | `+2.3473` nats | `[+2.3030, +2.3903]` |

Entropy-AUC wins/ties/losses were `500/0/0`. Mean final posterior entropy was `0.7477` for d2 and `3.0950` for d1; mean final MAP accuracy was 75.2% and 18.0%, respectively.

## Mechanism

- Every d1 trajectory began with an immediately informative weak screen.
- Every d2 trajectory began by activating panel A, which has exactly zero immediate EIG.
- d2 then chose posterior-dependent precise predicates and later switched to panel B when useful.
- Trial indices and hidden targets were paired, all selected actions were legal, and all inference was exact.
- The run made zero LLM calls.

## Interpretation

This is a deliberately constructed synthetic qualification, not external benchmark evidence. Its role is to establish a second, non-spatial task family where non-myopic planning is necessary and where policy quality can be verified exactly. It licenses a paid LLM proposal experiment; it does not by itself show that an LLM supplies useful search bias.

## Artifacts

- Frozen protocol: `results/nonmyopic/GATED_SENSOR_DEPTH_QUALIFICATION_PREREGISTRATION.md`
- Compact audit: `results/nonmyopic/gated_sensor_depth_qualification_20260722/REPORT.json`
- Generated report: `results/nonmyopic/gated_sensor_depth_qualification_20260722/REPORT.md`
- Full local traces: `results/nonmyopic/gated_sensor_depth_qualification_20260722/TRACES.jsonl.gz`
