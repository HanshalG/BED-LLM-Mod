# Number Game Fully Fresh Qwen Daily Source Result

Date: 2026-08-06

Run: `number-game-qwen-fully-fresh-daily-stages-20260806T000200Z`

## Decision

The fresh source stage is a **scientific gated null with a strong positive
non-myopic-versus-myopic component**. Every mechanics gate passed and dynamic
and fixed-support roots differed on `29/32` trees, so the hash-bound
history-blind control is authorized and must run on a later Europe/London
calendar day regardless of this source result.

The eventual composite cannot pass its full conjunction because the source
dynamic-versus-fixed comparison missed two frozen gates. The later control is
still required to complete the preregistered first-link mechanism test without
endpoint-dependent optional stopping.

## Source Science

| Comparison | Candidate Brier | Baseline Brier | Relative reduction | Tree W/T/L | Paired tree-bootstrap 95% CI | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Depth three vs myopic EIG | `.100505` | `.119288` | `15.75%` | `26/1/5` | `[-.026221,-.011693]` | all pass |
| Dynamic d3 vs fixed-support d3 | `.100505` | `.103457` | `2.85%` | `16/3/13` | `[-.007940,+.001551]` | reduction and CI fail |
| Depth three vs PTS | `.100505` | `.109076` | `7.86%` | `23` wins | `[-.012837,-.004445]` | descriptive |
| Depth three vs uniform random root | `.100505` | `.109547` | `8.25%` | `27` wins | `[-.012418,-.005583]` | descriptive |

All three frozen depth-three-versus-myopic gates pass: at least `8%` Brier
reduction, a difference interval below zero, and at least 20 tree wins.

For dynamic versus fixed support, the changed-root opportunity and 16-win
floors pass, but the `3%` reduction floor misses by `0.147` percentage points
and the interval crosses zero. Dynamic support also has worse mean Hamming by
`.002588` and lower canonical coverage by `.03125`; the latter interval is
entirely below zero. This is not an all-metric support improvement.

## Mechanics And Budget

- source trees / unique canonical targets: `32 / 33`;
- accepted requests / HTTP attempts: `3,680 / 3,680`;
- retries / provider-error retries / seed fallbacks: `0 / 0 / 0`;
- reasoning tokens / forced exits: `0 / 0`;
- all pooled initial, deployed retained, and 16-per-tree validation support
  floors passed;
- accepted-request cost: `$4.26043712`, below the tightened `$5.00` cap;
- authenticated balance after completion: `$27.702109737`;
- OpenRouter calls after the source stage: `0`.

The August 6 daily ledger opened at cumulative usage `$213.037453143` and
closed this paid block at `$217.297890263`, exactly matching the local accepted
request cost. No additional paid block is authorized for this day.

## Handoff

The source artifacts are bound by:

- source result: `13fd3361a8ef8f525f68733182a9bdb30151e37a4a5e14dfb35dde78540ab523`;
- source trees: `f812e4a356f5129a6f2f22d5f0f995b76b4ac624a0f312154064ff8c51f2c7b0`;
- source targets: `b799a5d6609f5e8088e2f0115ec1eaa3c4520cebcd187283daf679714cdc5e2b`;
- control authorization: `f75b038601191162e482c84ac8e7f559794ba3765cd4a9fb5239c8088ed09a2f`.

The authorization manifest reads only source mechanics and the changed-root
count. It records `run_control_regardless_of_source_science`; the first eligible
control launch is a later London date with a fresh daily ledger and the frozen
`$4.25` control cap.
