# INVALID-ENDPOINT: First Repaired Step 1 Attempt

This result is diagnostic only and must never be cited as policy evidence.

The run used the authorized canonical ten tasks, five rounds, seed 1304, and arms
naive non-thinking, naive thinking, and generation-thinking EIG. Automated metrics
reported zero final simulator inconsistency, but required manual transcript review
found a false terminal success on EIG task `customer_service:eval:0003`:

- Private solution: the dishwasher drain hose is clogged and the clog must be cleared.
- Selected action: `Straighten the drain hose behind the appliance.`
- Simulator reply: `Goal reached`.

Straightening a hose does not clear the stipulated clog. This is a simulator/private-
solution contradiction, so the endpoint is invalid under the mandatory faithfulness
gate. The entire paired result is quarantined.

For diagnostic context only, the automated read before manual rejection was:

| arm | resolution@5 | mean censored turns | coverage | requests | cost (USD) |
|---|---:|---:|---:|---:|---:|
| naive non-thinking | 0.30 | 4.80 | 1.000 | 202 | 0.0087 |
| naive thinking | 0.70 | 3.70 | 1.000 | 170 | 0.0794 |
| generation-thinking EIG | 0.30 | 4.90 | 0.952 | 4,624 | 0.5248 |

EIG versus naive non-thinking was 2 wins / 2 losses / 6 ties, but that comparison is
not a valid policy result. Raw run and combination directories remain ignored.
