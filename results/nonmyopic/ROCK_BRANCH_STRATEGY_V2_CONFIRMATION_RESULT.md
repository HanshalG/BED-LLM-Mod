# Rock Branch-Policy StrategyEIG Confirmation Result

The preregistered primary gate passed on both paper maps. Positive gains favor StrategyEIG.

| Map | Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |
| --- | --- | --- | --- | --- |
| 3-6 | Shared-roots d1 | +0.5205 [+0.4857, +0.5531] | +0.5192 [+0.3995, +0.6594] | 30/0/0 |
| 3-6 | Exhaustive d1 width | +0.5205 [+0.4855, +0.5537] | +0.5192 [+0.4008, +0.6613] | 30/0/0 |
| 3-6 | Random strategies | +0.3748 [+0.3229, +0.4267] | +0.3647 [+0.2822, +0.4504] | 30/0/0 |
| 5-7 | Shared-roots d1 | +0.4751 [+0.3929, +0.5544] | +0.4673 [+0.3701, +0.5616] | 30/0/0 |
| 5-7 | Exhaustive d1 width | +0.4517 [+0.3677, +0.5310] | +0.4095 [+0.2978, +0.5189] | 29/0/1 |
| 5-7 | Random strategies | +0.3622 [+0.2714, +0.4482] | +0.3564 [+0.2556, +0.4568] | 29/0/1 |

- Primary entropy-AUC gate: **True**.
- Truth-log-posterior corroboration: **True**.
- Run cost: `$0.32912333` across `1406` requests.
- Raw semantic rejects recovered: `26`; terminal failures: `0`.
- Logged horizon-1 item repairs: `153`.

## Mechanism

| Map | Arm | Movement decisions | Movement rate | Mean h2 exhaustive fraction |
| --- | --- | ---: | ---: | ---: |
| 3-6 | StrategyEIG | 140/240 | 0.583 | 0.906 |
| 3-6 | Exhaustive d2 | 150/240 | 0.625 | 1.000 |
| 3-6 | Shared-roots d1 | 0/240 | 0.000 | 0.430 |
| 3-6 | Exhaustive d1 width | 0/240 | 0.000 | 0.430 |
| 3-6 | Random strategies | 78/240 | 0.325 | 0.525 |
| 5-7 | StrategyEIG | 128/240 | 0.533 | 0.861 |
| 5-7 | Exhaustive d2 | 150/240 | 0.625 | 1.000 |
| 5-7 | Shared-roots d1 | 0/240 | 0.000 | 0.026 |
| 5-7 | Exhaustive d1 width | 0/240 | 0.000 | 0.132 |
| 5-7 | Random strategies | 82/240 | 0.342 | 0.333 |

StrategyEIG repeatedly selects zero-immediate-EIG movement that enables accurate future checks. The same-root d1 and exhaustive-d1 width controls never move. StrategyEIG also beats the matched random-policy prior, showing that proposal quality and non-myopic exact verification are both load-bearing.
