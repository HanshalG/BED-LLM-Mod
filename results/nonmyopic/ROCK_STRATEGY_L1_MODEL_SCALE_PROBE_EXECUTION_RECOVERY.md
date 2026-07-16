# Rock L1 Model-Scale Probe Execution Recovery

Registered on 2026-07-16 before resuming the one authorized Gemma 31B L1 formal probe.
This is an infrastructure recovery of the same formal attempt, not a second model,
prompt, grammar, seed, endpoint, or policy run.

## Interrupted Invocation

The registered command began with run ID
`nonmyopic-rock-strategy-l1-gemma31b-thinking-probe-20260716`, seed `12034`, and the
frozen configuration in
`configs/config_nonmyopic_rock_strategy_l1_gemma31b_thinking_openrouter.yaml`.
The local execution wrapper ended after 24 billed OpenRouter requests before the Python
runner wrote either `L1.json` or `L1_FAILURE.json`. No policy endpoint, partial trace,
candidate response, parser diagnosis, or gate value was produced or inspected.

The spend ledger is the sole surviving record of that fragment: 20,298 prompt tokens,
30,717 completion tokens, 21,763 reasoning tokens, and `$0.01731171`. A no-spend dry
run of the identical 60-trial/8-round/K=4/horizon-2 shape completes all 1,380 expected
LLM cells with the full mechanics suite passing. This rules out a deterministic runner
failure without observing a live policy outcome.

## Recovery Invariant

The recovery uses the same run ID, seed, model, thinking budget, prompt, grammar,
retry limit, maps, arms, exact scorer, paired observation schedule, concurrency, output
directory, and `$2.00` hard cap. Using the same run ID makes the adapter aggregate the
interrupted `$0.01731171` fragment and the recovery requests against one hard run cap.

The sole operational change is execution through a persistent PTY with a captured exit
status, so the formal process remains attached while requests are in flight. No model
output from the interrupted fragment is reused. If this recovery fails closed or is
interrupted again, the capability probe is recorded as unavailable and the program
consolidates without another launch.
