# Paprika Naive-Primary Arbitration Pilot

## Result

The final pre-registered arbitration variant passed its 10-task pilot gate under the
terminal-faithfulness-repaired endpoint.

| Arm | Resolution@5 | Mean censored turns | Coverage | Cost |
|---|---:|---:|---:|---:|
| Naive-primary arbitration | 0.60 | 3.7 | 96.97% | $0.291798 |
| Thinking naive | 0.40 | 4.9 | 97.67% | $0.115805 |
| Non-thinking naive | 0.30 | 4.8 | 92.68% | $0.010003 |

Against the pre-registered primary baseline, thinking naive, arbitration recorded 6
wins, 0 losses, and 4 ties. Mean paired censored-turn delta was -1.2 turns (95%
bootstrap CI [-2.2, -0.4]); resolution improved by 0.20. The frozen analyzer therefore
returns `arbitration_pass_stop_and_discuss`.

All 10 arbitration transcripts passed manual endpoint review. The arbitration arm had
zero structured parse failures, zero final simulator inconsistencies, and five literal
terminal claims/checks with zero terminal rejections. One additional non-literal
resolution occurred when the customer explicitly secured a loose trailer connector.

## Mechanism And Cost

EIG overrode the native default on 12/33 turns (36.4%). Four overrides immediately
selected the exact private remedy, but several overrides were unhelpful. The accepted
arbitration artifacts used 2,081 requests, 1,156,018 total tokens, and $0.291798, about
2.52x the cost and 9.42x the requests of thinking naive. Including the failed original
offset-8 attempt, the complete arbitration wave spent approximately $0.351868.

## Interpretation

This is a strong, endpoint-valid pilot signal in the direction required by the frozen
gate, not a definitive effect estimate. There are only 10 tasks, and the arbitration
prompt samples a fresh ordered candidate set, so the arm-level comparison mixes the
candidate-generation draw with EIG overrides. Per the pre-registration, no scaling or
new benchmark run is launched automatically. The project stops here for discussion.
