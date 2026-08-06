# Number Game Two-Draw Diversity Bonus Prospective Preregistration

Date frozen: 2026-08-06

## Purpose

Test whether diversity between the two independently seeded Qwen support
draws is useful planning signal rather than estimator noise. This is a new
prospective selector test on fresh trees. It does not alter the sealed August 7
history-blind control and cannot relabel any earlier result.

## Frozen Selector

For each candidate root, use only the six answer-conditioned future support
generations already required by retained-support depth-three planning: two
first-stage branches and four second-stage branches. At each branch, parse and
validity-filter both independent Qwen draws and compute Jaccard distance
between their canonical extension sets. The root diversity statistic is the
mean of those six distances.

Select the minimum of:

```text
z(within-tree cross-fitted depth-three predicted Brier)
- 0.5 * z(within-tree mean branch draw Jaccard distance)
```

Population standard deviation is used for both within-tree scales. A
zero-variance component contributes zero. Ties are broken by lower numeric
root. The shared initial generation is excluded because it cannot distinguish
roots. No realized target, endpoint, validation support, target coverage, or
history-blind control value enters selection.

The coefficient is exactly `-0.5`. There is no threshold or coefficient sweep
on the fresh cohort.

## Fresh Cohort

- 32 entirely new tree seeds, target seeds, and validation seeds;
- `qwen/qwen3.7-plus`, nonreasoning, for two support draws per planning
  history;
- the same strict item-isolated parser, retained-parent support, eight root
  candidates, cross-fitted depth-three risk, target bank, and exact-canonical
  endpoint used by the fully fresh source protocol;
- exact request accounting and the existing source mechanics floors;
- maximum paid cost `$5.00`, fitting one Europe/London daily allocation.

Seeds, hashes, and the concrete runner must be frozen before the first provider
request. No response, tree, target, or endpoint from the 96-tree development
cohort or August 6 fresh cohort may be reused.

## Primary Comparison

The headline comparison is the diversity-bonus depth-three selector versus
cross-fitted dynamic-support depth two on the same paired trees. This directly
tests the missing monotonic planning-horizon link:

1. exact-canonical endpoint Brier improves by at least 3%;
2. the tree-bootstrap 95% interval for bonus depth three minus depth two is
   entirely below zero; and
3. bonus depth three records at least 14 tree wins over depth two.

As a co-required mechanism check, the bonus must change at least 8 roots from
unadjusted depth three, its mean Brier must not be higher than unadjusted
depth three, and changed-root wins must exceed losses. The bonus-versus-
unadjusted interval is reported but is not required to exclude zero because
the retrospective paired effect is much smaller than the depth-three-versus-
depth-two effect.

The bootstrap uses 20,000 tree resamples with a seed fixed in the concrete
runner before execution.

## Secondary Comparisons

Report paired Brier comparisons against myopic EIG, compute-matched
fixed-support depth three, and unadjusted dynamic-support depth three. Also
report complete candidate-root ranking Spearman and candidate-set oracle
regret. These are secondary and cannot rescue a failed primary comparison.

## Interpretation Boundary

A pass would show that stochastic LLM hypothesis-generation diversity is a
useful non-myopic planning feature, strengthening the claim that the LLM's own
belief dynamics contain decision-relevant structure. It would not prove that
more draws always help, that Jaccard diversity is calibrated uncertainty, or
that the retrospective cohorts were confirmatory.

If the primary interval crosses zero or the direction reverses, close this
selector without coefficient tuning on the fresh cohort.
