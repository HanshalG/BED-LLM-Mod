# Rock Diagnosis LLM Pilot Retry Amendment

Registered: 2026-07-15, after the failed-closed first interface attempt and before
the replacement attempt.

## What Happened

The first registered attempt, run ID `nonmyopic-rock-diagnosis-llm-pilot-20260715`,
failed closed at trial 4, a width-expansion state at position `(0, 3)`. Gemma returned
the illegal action `move-WEST` twice although the prompt's legal set excluded it. The
strict parser halted the program; no endpoint metrics, comparison, or promotion read
was produced. The failure artifact records 447 requests, 166,513 prompt tokens, 8,202
completion tokens, zero reasoning tokens, and `$0.01921719` spend.

The original bounded retry resent an identical temperature-zero prompt, so the second
completion repeated the same invalid action. This is an interface failure, not a
scientific observation, and its partial trajectory is not used.

## Frozen Repair

All environment, seed, trajectories, horizon, candidate width, model, temperature,
token cap, exact scoring, controls, primary readout, and promotion criterion remain
unchanged. The only change is the existing one retry's content:

1. after a rejected completion, append that raw completion as an assistant message;
2. append an explicit validation message naming the error and restating the exact
   legal action IDs; and
3. request the same exact three-ID JSON object once more.

There is still no padding, action substitution, or repair by the experiment code. A
second invalid completion still fails closed. The replacement uses a fresh run ID and
output directory and does not reuse the failed partial trajectory. The extra projected
cost remains `$0.15`, and total exploration spend for this interface screen remains
far below `$1`.
