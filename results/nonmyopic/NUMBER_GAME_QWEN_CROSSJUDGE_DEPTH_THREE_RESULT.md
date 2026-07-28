# Number Game Qwen Cross-Judge Depth-Three Result

Date completed: 2026-07-28.

Status: **all frozen gates pass**.

## Design

This study holds all 64 policies from the two independent 32-tree
cross-fitted depth-three studies fixed. It changes only the hidden target
concept generator from Gemini 2.5 Flash to `qwen/qwen3.7-plus`.

Qwen generated 16 independent endpoint supports per tree, for 1,024 fresh
supports in total. It never saw a selected root, policy score, source
endpoint, or result, and no root was reselected. Qwen reasoning was explicitly
disabled.

The prerequisite blind serving gate also passed: 10/10 responses parsed to
21--24 valid unique rules with zero retries, reasoning tokens, or forced
exits, for `$0.0104736`.

## Primary Result

| Policy | Mean Brier | Mean Hamming | Mean coverage |
|---|---:|---:|---:|
| Cross-fitted depth three | 0.137662 | 0.051318 | - |
| Equally cross-fitted depth two | 0.140410 | 0.054361 | - |

Depth three improves Brier by `1.9573%`. Its mean paired difference is
`-0.002748`, with whole-tree bootstrap 95% CI
`[-0.004829, -0.000780]`. It wins 24 trees, ties on the 29 identical-root
trees, and loses 11.

Hamming improves by `5.5976%`, with difference CI
`[-0.005209, -0.000998]` and 25 tree wins. Exact-extension coverage rises
`1.1917` percentage points.

Novel-target Brier improves by `0.003874`, Hamming by `0.006073`, and
coverage by `2.4907` percentage points.

## Independent Source Blocks

Both source studies are directionally positive without selective pooling:

| Fixed policy block | Depth-three Brier | Depth-two Brier | Relative gain | 95% difference CI |
|---|---:|---:|---:|---:|
| First 32-tree study | 0.136301 | 0.138379 | 1.50% | [-0.004360, 0.000381] |
| Wholly fresh 32-tree study | 0.139023 | 0.142441 | 2.40% | [-0.006884, -0.000340] |

The first block remains directional with its interval crossing zero; the
second block independently excludes zero. The preregistered primary is the
pooled 64-tree comparison.

## Ranking And Controls

Qwen endpoint risk remains much more faithfully ranked by depth-three
cross-fitted source risk:

- depth-three Spearman: `0.7842`, bootstrap `[0.7426, 0.8229]`;
- depth-two Spearman: `0.4658`, bootstrap `[0.3895, 0.5406]`;
- depth-three concordance: `0.8326`;
- depth-two concordance: `0.6836`.

The fixed depth-three policy also beats:

- myopic EIG by `9.8743%`, CI `[-0.018522, -0.011712]`;
- fixed-support depth three by `6.7347%`, CI
  `[-0.013961, -0.006094]`; and
- PTS by `7.6632%`, CI `[-0.014180, -0.008800]`.

## Mechanics And Accounting

- accepted responses / HTTP attempts: `1,024 / 1,024`;
- retries / provider-error retries: `0 / 0`;
- reasoning tokens / forced exits: `0 / 0`;
- cost: `$1.06953856`;
- minimum valid rules in any endpoint support: `18`;
- minimum novel hypotheses across any tree's 16 draws: `130`.

Public artifact hashes:

- `RESULT.json`:
  `a7e0549f2f9ff7b1c2394ebe076bdbf1b70799479a3597e01bf66b7003edc1cb`;
- `ENDPOINTS.json`:
  `647de3c6561ff917691dc3c14176dc4007f90b17230bbb6ee690468d978a613d`;
- private raw responses, retained locally and not committed:
  `c32a090ebfdd91dd6553e0c8ae26bd4fa368d4c2982762e3a6d9b2bd5b8b0da4`.

## Interpretation

The monotonic depth effect is not specific to the Gemini target-concept
distribution. With every policy fixed, a third model family reproduces lower
held-out Brier and Hamming, higher coverage, favorable novel-target behavior,
and the depth-three ranking advantage.

This establishes target-model-family robustness. It does not establish a
third planning-generator family, a new task domain, or a third independent
policy-selection replication.
