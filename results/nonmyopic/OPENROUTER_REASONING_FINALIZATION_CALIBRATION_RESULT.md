# OpenRouter Reasoning Finalization Calibration

Endpoint-free live calibration of the generic forced-final adapter path after
the hierarchical h4 line was closed.

## Result

Two Gemma 4 26B thinking calls were used:

1. An ordinary 160-token combined allowance returned exact `{"ok":true}`
   natively in one request.
2. An intentionally tiny 33-token combined allowance length-stopped after
   reasoning. The adapter made one bounded 32-token non-reasoning continuation
   and returned exact `{"target":4}`.

The forced calibration recorded:

- forced exits: 1;
- forced-final requests/successes: 1/1;
- physical requests: 2;
- prompt/completion/reasoning tokens: 155/39/24;
- cost: `$0.00003517`.

Together with focused synthetic tests for the literal Wafer truncation notice
and native `reasoning_effort`, this validates the repaired generic serving path.
No Rock belief, policy endpoint, or closed h4 seed was used or rerun.
