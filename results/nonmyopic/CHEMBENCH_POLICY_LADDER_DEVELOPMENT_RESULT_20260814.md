# ChemBench Policy-Ladder Development Result

Date: 2026-08-14 (Europe/London)

This zero-call rehearsal uses only the already-open easy/medium/hard v3 source
states. It validates implementation and ranking mechanics; it is not the sealed
v4 result and authorizes no LLM call.

## Result

| Policy level | Aggregate terminal MSE | Successive reduction |
| --- | ---: | ---: |
| d1 | 0.0365256119 | - |
| d2 | 0.0325612802 | 10.8536% |
| d3 | 0.0246198189 | 24.3893% |

- d2 versus d1: 37 wins, 83 practical ties, 24 losses.
- d3 versus d2: 36 wins, 92 practical ties, 16 losses.
- d2 changes the d1 root on two of three slices.
- d3 changes the d2 root on all three slices.
- Planned prior terminal risk equals uniform truth-conditional replay to
  floating-point precision on every slice and level.
- The call-matched d1 replay adds zero oracle-producer misses.
- The separate banked-proposer verifier reproduces the complete result and
  gate exactly.

All frozen development conditions pass.

## Artifacts

- Result SHA256:
  `dfba49793b72c0d2955e61c4291e06daf870024f794ee04fbdda978a95d72c33`
- Transition bank SHA256:
  `6acb72eb7dd57af919243c7997ff786f3a21db4c00ef26db929236669024055b`
- Verification SHA256:
  `774f69d979c6420abdeab8ee255bbc3e9fd745e57537bfa060d86636831700dc`

The result supports running the prospectively frozen v4 zero-call mechanics
only after its implementation, protocol, producer, verifier, and tests are
committed and pushed. It does not establish LLM proposal calibration or
publishable efficacy.
