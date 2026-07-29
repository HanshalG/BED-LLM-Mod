# Number Game Qwen Pooled Second-Refresh Confirmation-32 Result

Run: `number-game-qwen-pooled-second-refresh-confirmation32-20260729T104705Z`

Status: **gated null**. The preregistered regeneration-specific primary
failed, although every mechanics gate and every required depth-three versus
myopic policy-efficacy gate passed.

## Prospective Mechanism Primary

Merged retained-plus-regenerated second-step support was compared with
parent-only second-step support on the same 32 fresh trees, first-step
supports, branch queries, eight cross-fit validation draws, and exact
33-concept endpoint.

- Roots differ on `19/32` trees.
- Merged-support Brier: `0.102677`.
- Parent-only Brier: `0.104123`.
- Relative Brier reduction: `1.39%`.
- Paired tree-bootstrap difference:
  `[-0.0055654, 0.0024412]`.
- Wins/ties/losses: `8/13/11`; wins minus losses: `-3`.

Only the root-difference gate passes. The minimum two-percent improvement,
interval-below-zero, and wins-minus-losses gates fail. The retrospective
`7.11%` second-refresh effect therefore does not replicate prospectively.

Merged support does beat generated-only support by `3.08%`, with paired
interval `[-0.0065995, -0.0002467]`, roots differing on `22/32`, and
`13/10/9` wins/ties/losses. This establishes that retaining compatible parent
hypotheses is important; it does not independently establish that the new
second-step generation adds value beyond parent-only support.

## Policy Efficacy

The full pooled depth-three policy strongly beats myopic EIG:

- Brier: `0.102677` versus `0.117810`, a `12.85%` reduction;
- paired Brier difference: `[-0.0217239, -0.0091635]`;
- Brier wins: `24/32`;
- Hamming: `0.021518` versus `0.026562`, an `18.99%` reduction;
- paired Hamming difference: `[-0.0084008, -0.0019127]`;
- Hamming wins: `22/32`;
- exact-target coverage difference: `+0.03598`.

All three preregistered myopic policy gates pass. Depth three versus
cross-fit depth two is directionally positive (`2.96%`) but its paired
interval `[-0.0077695, 0.0013413]` crosses zero.

## First-Link Diagnostic

Depth three changes the myopic root on `29/32` trees. On changed roots, mean
realized advantage is `0.01670`, with bootstrap interval
`[0.01028, 0.02367]` and `24/5` wins/losses. Predicted-versus-realized
Spearman correlation is `0.389`, but its interval
`[-0.0054, 0.7036]` narrowly crosses zero.

## Mechanics

- Exactly `32` fresh trees and `3,424` accepted requests.
- `3,434` HTTP attempts and `10` transparent retries, within the cap.
- All `3,424` provider draws are strict JSON; zero item salvage.
- Zero reasoning tokens and zero forced exits.
- Minimum deployed support sizes: initial `27`, retained first `15`,
  retained second `12`, and validation `20`.
- Run cost: `$4.10783796`.

All preregistered mechanics gates pass. Generated-only minima are diagnostic,
not deployed-support gates.

## Interpretation

The result independently replicates the average non-myopic-over-myopic policy
effect under clean mechanics. It does not replicate the sharper claim that
newly regenerated second-step hypotheses improve the policy beyond retained
parent support. The defensible conclusion is therefore that pooled,
path-conditioned depth-three planning works on this endpoint, while the
specific causal contribution of second-refresh generation remains
unconfirmed and retention is necessary.

Public artifact SHA-256:

- `RESULT.json`:
  `71281c483297e7ee4cc011d0ba795dabd28ec2d8598366771e11afee2af2a459`
- `TREES.json`:
  `5061327820199d1b27fac36e351708009184a7a0946bfa0ed21201b583c2f709`
- `TARGETS.json`:
  `f8e848dff267c6240003ec80d0832670434157a9fb7aef8560c3a9d59ad88c59`

Private raw-response SHA-256:
`1571d6555943e3f2e1256690afaeb2086e060366edbc3c7167165e6ac027456f`.
