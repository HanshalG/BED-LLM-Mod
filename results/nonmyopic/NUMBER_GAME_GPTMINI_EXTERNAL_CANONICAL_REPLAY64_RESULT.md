# Number Game GPT-5.4 Mini External-Canonical Replay64 Result

Date: 2026-07-29

Status: **formal monotonic-depth gated null; strong external nonmyopic versus
myopic component result**.

## Frozen Replay

- preregistration commit: `584525f`
- run:
  `number-game-gptmini-external-canonical-replay64-20260729T061141Z`
- result SHA-256:
  `33263d27fa0f3fb19b44908070df1a21fb014ffbf1bc1cd715d362d3ac531427`
- canonical-target SHA-256:
  `8e09b3910eece344b7486493df0980af2a3f941c17ad93ff016b7456da03892f`
- model calls / cost: `0` / `$0`

The replay scores 64 fixed GPT-5.4-Mini planning trees from two independent
32-tree studies on all 33 canonical Tenenbaum--Griffiths concepts. Policies
were fixed before this bank was combined with them. Compatibility and utility
are exact Boolean computations; no endpoint LLM or semantic judge is used.

## Primary Results

| Comparison | D3 Brier | Baseline | Gain | Stratified 95% CI | W/T/L |
|---|---:|---:|---:|---:|---:|
| Cross-fitted d2 | `0.108114` | `0.109298` | `1.08%` | `[-0.004520, 0.002120]` | 19/29/16 |
| Myopic EIG | `0.108114` | `0.124175` | `12.93%` | `[-0.021971, -0.010399]` | 45/3/16 |

Depth-three and depth-two roots differ on 35/64 trees. Both source blocks are
directionally nonnegative, but heterogeneously:

- study A: `0.108048` versus `0.108056`, a `0.0075%` gain;
- study B: `0.108180` versus `0.110540`, a `2.13%` gain.

The frozen d3-versus-d2 CI and 24-win gates fail. The 1% magnitude gate passes,
but the effect is not resolved on this deterministic bank. All myopic gates
pass comfortably.

## Diagnostics

All six prespecified diagnostics are favorable:

- Hamming improves by `0.002480` and coverage rises `0.331` percentage
  points versus d2, although their intervals are not both strictly favorable;
- d3 beats fixed-support d3 by `7.13%`, CI
  `[-0.013535, -0.003456]`;
- d3 beats PTS by `8.23%`, CI `[-0.013315, -0.006088]`;
- d3 beats random by `9.51%`, CI `[-0.014115, -0.008482]`;
- d3 rank Spearman is `0.503` versus `0.346` for d2; and
- pairwise concordance is `0.701` versus `0.631`.

## Interpretation

The exact external bank strongly supports non-myopic planning over myopic EIG,
including against fixed-support and randomized controls. It does not provide
a second resolved GPT-5.4-Mini monotonic-depth result: one 32-tree block is
essentially null, and pooled uncertainty crosses zero.

Together with the separately fresh Qwen canonical confirmation, this narrows
the evidence. The d3-over-d2 effect is positive under fresh Qwen planning and
LLM-generated endpoint studies, but is heterogeneous under the larger
deterministic GPT-Mini replay. The robust claim is that planning over
path-dependent LLM proposal dynamics beats myopic selection; monotonic benefit
from an additional planning step remains model and target-distribution
dependent.
