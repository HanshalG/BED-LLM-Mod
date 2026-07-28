# Number Game First-Retention Two-Query Audit

Date: 2026-07-28

Status: **completed post-hoc zero-call audit**

## Question

The first-retention path audit shows that retaining consistent initial
particles changes most future queries and recovers independent-target support.
This audit separates two possible two-query effects:

1. **Complete policy effect:** retained-root selection deployed on retained
   first support versus generated-root selection deployed on generated-only
   first support.
2. **Selection-only effect:** retained-root versus generated-root selection,
   with both roots deployed on the same retained first support.

The first comparison measures the full belief-transition intervention. The
second asks whether its source-particle risk scores also choose a better root.

## Exact Method

For every tree in the open eight-tree depth-three development dataset and
independent fresh six-tree confirmation:

- reconstruct generated-only and first-retained branch supports;
- minimize source-particle two-query posterior-predictive Brier risk over the
  same eight roots under each support transition;
- deploy each policy for two exact deterministic queries on the independently
  generated target rules; and
- compare target Brier, best-rule Hamming error, and exact-extension coverage.

For the selection-only comparison, both selected roots use retained supports.
Whole trees are the unit for 50,000-sample bootstrap intervals. There are no
model calls.

## Results

### Complete belief-transition intervention

| Source | Trees | Retained Brier | Generated Brier | Reduction | Wins |
|---|---:|---:|---:|---:|---:|
| Development | 8 | .18018 | .19671 | 8.40% | 8/8 |
| Fresh confirmation | 6 | .18117 | .19053 | 4.91% | 6/6 |
| Combined | 14 | .18061 | .19406 | 6.94% | 14/14 |

The combined mean Brier difference is `-0.01346`, with tree-bootstrap interval
`[-0.01742, -0.00959]`. Mean Hamming falls by `0.05650`
(`[-0.06760, -0.04512]`) and exact-extension coverage rises by `28.18`
percentage points (`[22.20, 34.29]`). Both datasets are directionally
identical and individually have intervals excluding zero.

### Root-selection component

Retained and generated source-risk selectors choose different roots on 7/14
trees. Under the common retained-support endpoint, the retained selector
reduces Brier by only `0.64%`: mean difference `-0.00116`, interval
`[-0.00452, 0.00182]`, with 3 wins, 4 losses, and 7 ties. Hamming is
effectively unchanged and coverage is 1.60 points lower on average.

Mean source-risk versus independent-target Brier Spearman is `0.488` for
generated-only scoring and `0.474` for retained scoring. Thus first retention
does not improve root-ranking fidelity in these open trees.

## Interpretation

First retention is a strong inference intervention: preserving valid particles
substantially improves the complete two-query endpoint on every tree. The gain
does not come from consistently choosing a better root. It comes from the
retained LLM belief transition itself.

This sharpens the status of the frozen depth-three experiment. Its candidate
and depth-two baseline share retained first supports, so the established
two-query support gain cannot by itself make depth three win. A positive result
must come from the second regenerated transition and its non-myopic root
ranking. The current audit is therefore mechanism evidence, not a substitute
for the fresh policy test.

## Integrity

- Development `TREES.json` SHA-256:
  `016ed9218e9f034984e0745c9a7cb62b4111901db821c065e20c38c5ce76b65f`
- Fresh confirmation `TREES.json` SHA-256:
  `b6c157f7b4d15b4934a2329adeba7c0c28557293aab48de201f5ec8aba0b4f3c`
- Audit `RESULT.json` SHA-256:
  `d467dfc523dba8193eab041f8f2a717adbf18ad8b8dc75e537ce3b62b8fd024c`
- Model calls and cost: `0`

Artifact:
`results/nonmyopic/number_game_first_retention_two_query_audit/RESULT.json`
