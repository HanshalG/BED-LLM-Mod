# Number Game Qwen History-Blind Matched-32 V3 Result

Date completed: 2026-07-30

## Decision

**The formal result is a conjunctive gated null with strong direct evidence
that answer-conditioned generation improves the second-stage belief state.**

Three of four preregistered scientific gates pass. The failed gate asks
whether the prompt-conditioning benefit is larger on average at the dynamic
root than at the compute-matched fixed-support root. Its point estimate is
positive, but its confidence interval crosses zero. The source
mechanics-qualified status is unchanged.

## Matched Support Result

The control uses two fresh Qwen3.7 Plus nonreasoning generations from the
initial no-observation prompt for each stored branch slot. Both arms then use
the same consistency filter and recursive parent retention.

After the first answer:

- conditional predictive MSE: `0.0380216`;
- history-blind predictive MSE: `0.0207735`;
- conditional-minus-blind difference: `+0.0172481`;
- conditional truth coverage is `0.5411` versus `0.4364`, or `+10.46` points.

After the second answer:

- conditional predictive MSE: `0.0325153`;
- history-blind predictive MSE: `0.0380303`;
- difference: `-0.0055150`, with 95% interval
  `[-0.0078411, -0.0031118]`;
- conditional truth coverage is `0.6116` versus `0.4458`, a `+16.58` point
  difference with interval `[15.61, 17.58]` points.

Answer conditioning therefore initially trades calibration for coverage, then
produces a better calibrated and much better covering support after the second
observation. This is a direct path-dependent belief-dynamics effect; it is not
explained by re-filtering the same model call.

## Selected-Root Gate

Dynamic and fixed-support depth-three policies choose different roots on
`27/32` trees, passing the minimum-20 gate.

- mean conditioning benefit at the dynamic root: `0.0097489`;
- mean benefit at the fixed root: `0.0084542`;
- dynamic-minus-fixed contrast: `+0.0012947`;
- contrast interval: `[-0.0068939, +0.0096250]`.

The registered positive-mean-contrast gate fails. Descriptively, the
tree-level contrast strongly predicts realized dynamic-root advantage:
Spearman `0.7167`, with interval `[0.3931, 0.8889]`. Thus conditioning quality
tracks where dynamic selection helps, but this cohort does not establish that
the selected dynamic roots receive a larger average benefit.

## Mechanics And Cost

- exactly `32` trees, `1,536` branch slots, and `3,072` accepted requests;
- exactly `3,072` HTTP attempts, with zero retries or provider retries;
- all draws strict JSON and at least `16` valid rules;
- every pooled support has at least `24` valid rules;
- zero reasoning tokens and zero forced exits;
- second-draw novelty range `2..24`, mean `7.7526`, descriptive only;
- cost `$3.25718784`, below the frozen `$4.25` cap.

No V1 or V2 response was reused. Both prior endpoints remained unaccessed.
An independent zero-call replay from the saved controls exactly reproduced all
32 scored trees and the full 20,000-sample bootstrap.

## Artifacts

Run:

`results/nonmyopic/number_game_qwen_history_blind_matched32_v3/number-game-qwen-history-blind-matched32-v3-20260730T173000Z`

- `RESULT.json` SHA256:
  `29bb76074dfb53a6efef352fde88fbca6c3a7dc591d0936d132c82050f1dc71d`
- `CONTROLS.json` SHA256:
  `6db0e3e272cb843250174a2252098c4a88d3d4a154d3287ba5542f9d2eb09a1b`
- private raw responses SHA256:
  `bb2c9c90da0458b1c6d04eb0505c67e91e7619c1d950fd4c9659069901d32266`
- run log SHA256:
  `9c37b27d3bfdfb2573eb3c4af013f0ad22a97d3413f708a20b596d7ddc903199`

Authenticated balance after the run was `$5.749374309`.
