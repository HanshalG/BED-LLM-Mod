# Range-Gated RockSample[7,8] Exact Depth-Four Result

The fresh seed-24201 producer and independent audit **pass every frozen gate**.

## Paired Result

| Endpoint | Mean d4-over-d3 gain | Producer 95% CI | Independent 95% CI | W/T/L |
| --- | ---: | ---: | ---: | ---: |
| Entropy AUC | +0.348436 | [+0.348436, +0.348436] | [+0.348436, +0.348436] | 500/0/0 |
| Truth-log-posterior AUC | +0.342417 | [+0.302366, +0.380107] | [+0.302054, +0.380858] | 470/0/30 |

## Mechanism

Only the standard-map start changes, from `(0,3)` to `(6,6)`. Rock 4 remains at
`(6,3)`, three moves away.

- Exact d3 starts with `check-0` in all 500 trials and never reaches an on-site
  inspection within the eight-round policy.
- Exact d4 takes `move-NORTH` in rounds 1--3 and `check-4` on site in round 4 in
  all 500 trials.
- D4 then starts another enabling route, taking
  `move-NORTH, move-NORTH, move-WEST, check-1` in rounds 5--8.

The entropy-AUC paired difference is identical across truths because entropy
dynamics depend on the action/observation channel and this policy uses the same
load-bearing route in every trial. Truth-log gains vary with the realized rock
states and observations but retain a strongly positive independent interval.

## Audit

The independent implementation replayed all 1,000 eight-round traces and:

- independently re-solved every d3/d4 action;
- matched every planning value and immediate EIG;
- reproduced every position and common-random observation;
- matched every posterior entropy and truth-log value;
- reproduced both aggregate means; and
- obtained fresh positive bootstrap intervals.

This is exact structural evidence that a fourth planning step can be load-bearing.
It is not yet an LLM h4 policy result. A proposal interface requires a separate
preregistration.
