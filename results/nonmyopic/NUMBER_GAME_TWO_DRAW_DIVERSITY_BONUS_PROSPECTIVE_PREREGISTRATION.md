# Number Game Two-Draw Diversity Bonus Prospective Preregistration

Date initially frozen: 2026-08-06

Power amendment frozen: 2026-08-06, before any prospective seed was opened.
The hash-bound retrospective power audit recommended 64 rather than 32 trees;
its result SHA-256 is
`3dfd1605deba87641c791718a8b50d81e971062fee85612fa94638fc674f156e`.

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

Use 64 entirely new trees in two mandatory blocks:

- Block A tree seeds `110000..110031`, target seeds `110100..110131`,
  validation seeds beginning at `110200`, and source bootstrap seed `110800`;
- Block B tree seeds `111000..111031`, target seeds `111100..111131`,
  validation seeds beginning at `111200`, and source bootstrap seed `111800`;
- combined analysis bootstrap seed `112800`;
- `qwen/qwen3.7-plus`, nonreasoning, for two support draws per planning
  history;
- the same strict item-isolated parser, retained-parent support, eight root
  candidates, cross-fitted depth-three risk, target bank, and exact-canonical
  endpoint used by the fully fresh source protocol;
- exact request accounting and the existing source mechanics floors;
- exactly `3,680` accepted requests per block, `7,360` total;
- maximum paid cost `$5.00` per block on separate Europe/London days,
  `$10.00` total.

Seeds, hashes, and the concrete runner must be frozen before the first provider
request. No response, tree, target, or endpoint from the 96-tree development
cohort or August 6 fresh cohort may be reused.

The concrete runner is
`scripts/number_game_two_draw_diversity_bonus_confirmation64_staged.py`.
Each block requires a current Europe/London ledger with the full `$5.00`
allowance remaining, checks live account-wide usage and balance before source
construction, and reconciles measured and posted spend. It authorizes no
second confirmation block that day. After reconciliation, a separately frozen
experiment unrelated to this confirmation may use only the exact remaining
daily allowance. Such a tail block cannot use these confirmation seeds,
artifacts, mechanics, or scientific values for selection, and cannot affect
Block B authorization or the combined analysis.

Block A produces a mechanics-only authorization record. Block B runs on a
strictly later London date whenever all Block A mechanics pass, regardless of
Block A policy comparisons, selected roots, or scientific values. There is no
scientific stopping or human decision between blocks. If Block A mechanics
fail, Block B is forbidden and the experiment is `mechanics_failed`. The
combined 64-tree endpoint is the sole confirmatory decision.

## Primary Comparison

The headline comparison is the diversity-bonus depth-three selector versus
cross-fitted dynamic-support depth two on the same paired trees. This directly
tests the missing monotonic planning-horizon link:

1. exact-canonical endpoint Brier improves by at least 3%;
2. the tree-bootstrap 95% interval for bonus depth three minus depth two is
   entirely below zero; and
3. bonus depth-three tree wins exceed losses against depth two.

As a co-required conservative check, the bonus must change at least 16 of 64
roots from unadjusted depth three and its mean Brier must not be higher than
unadjusted depth three. Bonus-versus-unadjusted wins/losses and interval are
reported but are not additional gates because the paired mean already defines
non-worsening and many trees tie.

The combined bootstrap uses 20,000 resamples of the 64 paired tree rows with
seed `112800`.

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
