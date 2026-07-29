# Number Game Pooled-Margin Scale Analysis Plan

Date frozen: 2026-07-29, after the pooled-support confirmation result was
known and before computing any transformed first-link score.

This is a retrospective zero-call mechanism diagnostic. It cannot rescue,
replace, or reclassify the source `gated_null`.

## Motivation

The fresh pooled policy improves average Brier over myopic, its selected roots
have positive mean realized advantage, and its within-tree root ranking is
positive. However, the raw simulated advantage magnitude has null cross-tree
Spearman. Pooled supports vary in size and diversity, so raw Brier-risk
margins may have tree-specific scales.

## Single Frozen Transformation

For each of the 30 changed-root trees:

1. compute the raw predicted advantage between the myopic and selected
   depth-three roots;
2. compute the population standard deviation of all eight cross-fitted
   depth-three candidate-root risks in that tree;
3. divide the raw advantage by that standard deviation.

No alternative scale, clipping, rank transform, threshold, subset, or
regularizer will be tried.

Report raw and normalized Spearman with realized exact-endpoint advantage,
plus their paired difference, using 20,000 tree-bootstrap samples with seed
`63700`. Also report the scale distribution and its correlation with realized
advantage.

The fixed diagnostic is positive only if normalized Spearman is at least
`0.25` and its bootstrap lower bound is above zero. Otherwise this
scale-normalization explanation closes. Model calls and cost are zero.
