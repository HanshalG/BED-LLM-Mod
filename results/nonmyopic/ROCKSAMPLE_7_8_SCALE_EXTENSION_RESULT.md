# RockSample[7,8] StrategyEIG Scale Extension

Both preregistered stages passed on the canonical eight-rock geometry.

## Exact Qualification

Across 1,000 paired trajectories, exhaustive d2 reduced final entropy by `+1.6775` nats relative to exhaustive d1 (95% CI `[+1.6568, +1.6980]`; W/T/L `1000/0/0`).

## LLM Strategy Confirmation

Positive paired gains favor StrategyEIG.

| Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |
| --- | --- | --- | --- |
| Shared-roots d1 | +0.7502 [+0.7050, +0.7938] | +0.6622 [+0.6029, +0.7206] | 30/0/0 |
| Exhaustive d1 width | +0.7297 [+0.6874, +0.7724] | +0.6511 [+0.5813, +0.7303] | 30/0/0 |
| Random strategies | +0.6735 [+0.6251, +0.7212] | +0.5841 [+0.5158, +0.6554] | 30/0/0 |

- Primary entropy-AUC gate: **True**.
- Truth-log corroboration: **True**.
- Cumulative serving cost: `$0.21575537` across `904` requests.
- Reused accepted cells: `856`; raw rejected responses retained: `34`.

## Mechanism

| Arm | Movement decisions | Movement rate | Mean h2 exhaustive fraction |
| --- | ---: | ---: | ---: |
| StrategyEIG | 193/300 | 0.643 | 0.858 |
| Exhaustive d2 | 210/300 | 0.700 | 1.000 |
| Shared-roots d1 | 0/300 | 0.000 | 0.064 |
| Exhaustive d1 width | 0/300 | 0.000 | 0.133 |
| Random strategies | 104/300 | 0.347 | 0.269 |

StrategyEIG again spends zero-immediate-information actions on movement before checking. Shared-roots d1 and exhaustive d1 width cannot value that enabling action, while random branch policies retain the verifier but remove the LLM search prior.
