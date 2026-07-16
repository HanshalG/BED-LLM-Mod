# Rock Strategy-Prior L1 Formal Retry Amendment

Recorded on 2026-07-16 after the first formal invocation failed closed and before a
replacement run. The failed invocation produced no `L1.json`, arm comparison, policy
endpoint, or gate result. Only its failure type, invalid-response classes, and usage
were inspected.

## Failure

The seed-12032 invocation stopped because a strategy cell failed the registered
move/check root-mix validator after its single feedback retry. Concurrent trials had
made 316 API requests before shutdown, using 303,655 prompt tokens, 170,928 completion
tokens, zero reasoning tokens, and `$0.09280504`.

Of 142 rejected responses, 120 had the same root-mix error. Inspection showed a
specific interface ambiguity: the model described strategies as immediate checks but
guarded `check_rock` with `at_rock`, then used `target_rock` as the fallback. At the
current position these compiled to movement roots. The prompt had not stated clearly
that Rock Diagnosis checks are legal remotely and become noisier with distance.

## One Authorized Repair

- State explicitly that `check_rock` is legal from every grid position and is the
  distance-dependent remote sensor.
- Require one strategy whose currently matching action is a direct check and one whose
  current action is movement. Show an unconditional `check_rock` fallback as the
  guaranteed construction.
- Include the compiled root-action list and that same construction in validation
  feedback.

There is no programmatic action insertion, candidate padding, parser fallback, reward
change, scorer change, or endpoint change. The exact dynamics and the move/check mix
required of both LLM and random cells remain unchanged.

The replacement is the protocol's sole Rock prompt/grammar rerun. It uses fresh seed
`12033`; all maps, arms, K=4, horizon 2, rounds, paired trial count, model,
non-thinking mode, temperature, retry count, concurrency, exact scorer, bootstrap,
gate, and `$1.00` run cap remain frozen. A second failure or a completed loss to the
random-strategy control triggers the registered pivot rather than another Rock rerun.
