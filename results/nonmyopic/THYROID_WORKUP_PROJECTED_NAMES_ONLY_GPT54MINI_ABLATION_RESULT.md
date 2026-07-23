# UCI Thyroid Projected Names-Only GPT-5.4 Mini Ablation Result

The fresh projected names-only ablation completed all trajectories but failed four
of six frozen scientific gates. Independent replay passed every mechanical check and
recomputed the failed scientific gate.

## Paired result

| Endpoint | Mean gain | Registered paired 95% CI | Independent 95% CI | W/T/L |
| --- | ---: | ---: | ---: | ---: |
| Entropy AUC vs exact d1 | +0.089037 | [+0.067381, +0.110521] | [+0.067116, +0.110865] | 37/9/4 |
| Truth-log AUC vs exact d1 | +0.020717 | [+0.012035, +0.028640] | [+0.012150, +0.028590] | 37/9/4 |
| Entropy AUC vs matched random | -0.012867 | [-0.049014, +0.025084] | [-0.051023, +0.024983] | 19/3/28 |
| Truth-log AUC vs matched random | -0.105416 | [-0.263922, -0.002346] | [-0.263147, -0.002372] | 17/3/30 |

Names-only projection recovered 43.15% of exhaustive depth two's entropy gain and
selected blood collection first on 0/50 trajectories. It passed both exact-d1 lower
bounds but failed both matched-random lower bounds, the 60% recovery gate, and the
75% collection gate.

Projection was negligible and passed both contribution ceilings: one of 350 logical
cells (0.286%) and two of 3,765 branches (0.0531%). The model requests contained no
expected-entropy or information-gain cards. The independent audit replayed all 1,600
arm decisions, 2,473 unique exact subtrees, and both projected branches.

## Factorial interpretation

The three completed fresh-seed GPT conditions form a descriptive mechanism table;
paired claims remain within each row's own controls.

| Interface | Entropy gain vs d1 | Entropy gain vs random | d2 recovery | Collection | Projected branches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Names only | +.0641 | -.0339 | 30.8% | 2/50 | 0 |
| Names only + projection | +.0890 | -.0129 | 43.2% | 0/50 | 2/3,765 |
| Utility cards + projection | +.2026 | +.1185 | 99.4% | 50/50 | 17/4,338 |

Projection alone fixes rare serving errors but does not recover the non-myopic setup
choice or beat random continuations. Calibrated branch-local utility is the
load-bearing quality intervention; bounded projection is the load-bearing serving
intervention. The positive result therefore depends on both machine-grounding
components, while GPT still authored more than 99.6% of returned branches.

S0 plus S1 used 372 physical requests, 676,678 prompt tokens, 41,148 completion
tokens, zero reasoning tokens or forced exits, and `$0.45559290`. Project spend is
`$36.64346041 / $110`, leaving `$73.35653959`.
