# Number Game Retained Depth-Three Confirmation Result

Date: 2026-07-28

Protocol:
`NUMBER_GAME_RETAINED_DEPTH_THREE_CONFIRMATION_PREREGISTRATION.md`

Run:
`results/nonmyopic/number_game_retained_depth_three_confirmation/number-game-retained-depth-three-confirmation-20260728T070902Z`

## Verdict

**Gated null with a directional monotonic depth effect.**

Retained rejuvenation eliminates the destructive second-refresh collapse and
the simulator remains positively ranked against independent targets. Mean
Brier, Hamming, and coverage all improve versus depth two. The frozen
conjunction nevertheless fails because depth three changes only 2/6 roots and
wins only those 2/6 trees, below the required three. Candidate and
generated-only roots differ on only 1/6, and one first-refresh branch contains
seven valid rules rather than the required eight.

No tree, threshold, or support was repaired, and the run was not repeated.

## Primary Comparison

| Control | Candidate Brier | Control Brier | Relative gain | Brier wins | Brier difference 95% CI | Hamming difference | Coverage difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| depth-two predictive risk | 0.18022 | 0.18363 | 1.86% | 2/6 | [-0.00981, 0.00000] | -0.00448 | +0.02207 |
| parent-only depth three | 0.18022 | 0.18255 | 1.28% | 2/6 | [-0.00961, 0.00332] | -0.00778 | +0.05995 |
| generated-only depth three | 0.18022 | 0.18459 | 2.37% | 1/6 | [-0.01311, 0.00000] | -0.00452 | +0.06522 |
| myopic EIG | 0.18022 | 0.19633 | 8.20% | 4/6 | [-0.04156, 0.00710] | -0.02226 | +0.03899 |
| fixed-support depth three | 0.18022 | 0.19892 | 9.40% | 5/6 | [-0.02849, -0.00854] | -0.03096 | +0.13382 |
| exact uniform random | 0.18022 | 0.18709 | 3.67% | 4/6 | [-0.01390, -0.00001] | -0.01049 | +0.04999 |
| positive-test strategy | 0.18022 | 0.17740 | -1.59% | 2/6 | [-0.00306, 0.00827] | -0.00282 | +0.02263 |

Candidate-minus-depth-two differences on targets absent from the initial
support are `-0.00254` Brier, `-0.00622` Hamming, and zero coverage change.

## Root-Level Diagnosis

Depth-three and depth-two roots are identical on four trees, producing exact
endpoint ties. The two changed roots both improve held-out Brier:

- seed `27800`: root `0` instead of `22`, Brier difference `-0.01921`;
- seed `27805`: root `40` instead of `54`, difference `-0.00123`.

Thus the directional aggregate is not a balance of wins and harmful
reversals. It is sparse: deeper planning finds an advantageous different root
on only two fresh trees.

Retained and parent-only roots differ on 3/6, with a `1.28%` mean Brier gain
for retained generation. Retained and generated-only roots differ on only
1/6. The second LLM refresh is useful relative to parent filtering, but the
retention repair rarely changes the root selected from generated-only scoring.

## Ranking And Mechanics

Mean within-tree source-risk/independent-target Brier Spearman is `0.532`
(`[0.310, 0.742]`) and pairwise concordance is `0.726`
(`[0.631, 0.821]`), passing both frozen ranking gates.

Every retained second support has at least nine hypotheses, including cells
where generated-only support has zero. The sole structural failure is seed
`27804`, whose minimum first-refresh support is seven. Extending retention to
the first refresh would change the second queries and requires new generated
branches; it cannot be evaluated faithfully from this run.

## Failed Gates

Four of the frozen conjunctive gates fail:

1. all initial and first supports valid: minimum first support `7 < 8`;
2. depth-three root differs from depth two on `2/6 < 3/6`;
3. Brier wins versus depth two are `2/6 < 3/6`; and
4. retained root differs from generated-only on `1/6 < 2/6`.

All other mechanics, direction, Hamming, coverage, parent-only, myopic, and
ranking gates pass.

## Accounting And Provenance

- Accepted responses / HTTP attempts: `300 / 300`
- Retries / provider-error retries: `0 / 0`
- Reasoning tokens / forced exits: `0 / 0`
- Reported cost: `$0.9743724`
- Live OpenRouter balance after completion: `$0.127224544`
- Result SHA-256:
  `39edaa698d789b1306f9a1978a5d86123eb79a684976948c66e7d7faaed1b914`
- Public tree SHA-256:
  `b6c157f7b4d15b4934a2329adeba7c0c28557293aab48de201f5ec8aba0b4f3c`
- Private raw-response SHA-256:
  `4d3a5f19d2088986e68f7754ca3338aee3039e9c7f3076187b12867785f91b5c`
- Run-log SHA-256:
  `6534cb7d939caf2e8966348fde8668e8215b6fc2de14624fb57d0024961c0baa`

The shell sidecar `tee` opened before its parent directory existed, so the
outer pipeline returned status 1 after the runner completed. The authoritative
runner produced complete `RESULT.json`, `TREES.json`, private checkpoint, and
300-row structured usage log. This was not a model or scientific failure.
