# Number Game Qwen History-Blind Matched-32 Failure Result

Date: 2026-07-30

## Status

**Failed closed before scientific scoring.** The run completed all 3,072
provider requests and parsed every response, but failed one frozen auxiliary
pool-diversity mechanic. No canonical endpoint was accessed, no scientific
aggregate was produced, and none of these responses may be reused.

## Passed Mechanics

- source trees: exact frozen `32`;
- branch slots: exact `1,536`;
- accepted requests / HTTP attempts: `3,072 / 3,072`;
- retries / provider-error retries: `0 / 0`;
- reasoning tokens / forced exits: `0 / 0`;
- strict JSON draws: `3,072 / 3,072`;
- valid unique rules per raw draw: `18..24`, all above `16`;
- pooled unique rules: `24..41`, all at or above `24`;
- run cost: `$3.21062528`, below the `$4.25` cap.

## Failed Mechanic

The frozen gate required every second draw to add at least two unique
extensions. Exactly `2/1,536` pools added only one:

- tree 20, `second:1:0:60:1`, pool size `25`, draw counts `[24,21]`;
- tree 23, `first:36:1`, pool size `24`, draw counts `[23,21]`.

The remaining `1,534/1,536` pools pass this threshold. This is not grounds to
waive the frozen conjunction. The run remains a mechanics failure.

## Decision

Bank the run permanently. A successor may use entirely fresh seeds and remove
the per-second-draw novelty threshold because final pooled size and per-draw
validity already enforce usable support. That change must be frozen before new
responses, and second-draw novelty must remain descriptive.

## Provenance

- public `FAILURE.json` SHA256:
  `f394c889a56fdb20854d28cda0fa2927ae1a8d40e02c54f0f400f3f2757fdcb3`
- uncommitted public-controls SHA256:
  `3696476ea4aabd17f92ba06fb5142411369c11cc9aabb91f0c2feeac25437f89`
- private raw-response SHA256:
  `e60fca6654e8407d5c6701a32b755aa5e84999f164889a59a9ccba9c64607822`
- source endpoint accessed: `false`
- model calls / accepted cost: `3072` / `$3.21062528`
