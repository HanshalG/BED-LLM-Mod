# Number Game DeepSeek V4 Flash Paired Efficacy32 Result

Date completed: 2026-07-30

## Decision

**Close the DeepSeek V4 Flash replacement route. Qwen3.7 Plus remains the
paper-critical and scaled Number Game planner.**

DeepSeek was inexpensive and its selected depth-three policy was statistically
non-inferior to the paired Qwen policy, but it did not pass the frozen
replacement protocol. The run is `mechanics_failed`, and its depth-three gain
over its own myopic policy missed two of three required intelligence gates.

## Frozen Efficacy Result

On the exact same 33 canonical targets and the exact same eight stored Gemini
validation supports per tree:

- DeepSeek depth-three Brier: `0.1022483`
- DeepSeek myopic Brier: `0.1065774`
- relative reduction: `4.0619%`
- paired tree-bootstrap difference: `[-0.007847, -0.000926]`
- tree wins/ties/losses: `18/6/8`

The direction is real, but the preregistered replacement minimum was at least
`8%`, a confidence interval below zero, and at least `20/32` wins. Only the
confidence-interval gate passed.

## Paired Qwen Comparison

- Qwen depth-three Brier: `0.1045638`
- DeepSeek minus Qwen mean Brier: `-0.0023155`
- paired 95% interval: `[-0.007104, 0.002169]`
- DeepSeek wins/losses: `15/17`
- frozen non-inferiority margin: `+0.005`

The interval upper bound is below the margin, so DeepSeek passes the Qwen
non-inferiority gate. This does not override the failed own-myopic intelligence
and mechanics gates.

## Mechanics And Cost

- exactly `32` complete trees and `1,568` accepted planner requests
- no target-generation or validation-generation provider calls
- all `49` response shapes per tree parsed under the unchanged strict parser
- zero reasoning tokens and zero forced exits
- `2` provider-error retries, failing the frozen zero-provider-error gate
- four trees had a merged first-branch minimum below `8`: minima `5`, `4`,
  `7`, and `6`
- all initial supports passed; all merged second branches passed
- run cost: `$0.3496433702`, below the `$0.50` cap

The successful run used four concurrent trees, at most `128` planner requests
in flight. A prior serial infrastructure attempt was aborted before scoring
after one tree because long-tail provider latency projected roughly four hours;
it is not an efficacy result.

## Interpretation

Artificial Analysis was useful for identifying DeepSeek as a cheap candidate,
but generic reasoning scores did not predict the deployed nonreasoning planner
role. In this role, conditional support coverage and the incremental value of
non-myopic planning matter more than broad benchmark score. DeepSeek is a valid
low-cost exploratory generator, but the frozen result does not justify replacing
Qwen for paper-critical planning.

## Artifacts

Successful run:

`results/nonmyopic/number_game_deepseek_v4_flash_paired_efficacy32/number-game-deepseek-v4-flash-paired-efficacy32-20260729T230558Z`

- `RESULT.json` SHA256:
  `add44efc18dc72121fcffb264ec88836f75d9b36b6a251a9b3a880446c460330`
- `TREES.json` SHA256:
  `0b678c0d542ce91f003036acbec1e9920222edea8df8cc8a294a52e917189fda`
- private raw responses SHA256:
  `9ff345e60968e256dcd2a7210f7922b650a909549f4e3698348ddc917233af78`

No threshold repair, support-minimum relaxation, parser change, subset analysis,
or rerun is authorized by this result.
