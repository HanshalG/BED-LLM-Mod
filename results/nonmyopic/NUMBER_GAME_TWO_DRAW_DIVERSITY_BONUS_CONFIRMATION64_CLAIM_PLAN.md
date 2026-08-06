# Number Game Diversity Confirmation Claim Plan

Frozen: 2026-08-06, before either prospective block or any seed in the
64-tree confirmation was opened.

## Purpose

Keep the prospective policy result separate from causal evidence about the
LLM's stochastic, path-dependent hypothesis generation. The original
preregistration and scientific status remain unchanged. This document only
tightens what may be claimed from each possible result.

## Evidence Families

### Registered non-myopic depth

All three frozen bonus-depth-three versus dynamic-depth-two gates must pass:

- at least 3% paired Brier reduction;
- the paired tree-bootstrap 95% interval is entirely below zero; and
- tree wins exceed losses.

This establishes a prospective planning-depth result for the deployed
diversity-aware policy. It does not by itself show that the diversity bonus
caused the gain, because unadjusted depth three may perform similarly.

### Registered selector viability

Both frozen bonus-versus-unadjusted checks must pass:

- at least 16 of 64 roots change; and
- mean bonus Brier is not worse than unadjusted depth three.

This shows that the bonus is behaviorally active and viable. Non-worsening is
not superiority and must not be described as a causal selector improvement.

### Diversity-selector superiority

The paired bonus-versus-unadjusted depth-three comparison must satisfy all of:

- mean bonus-minus-unadjusted Brier is strictly below zero;
- its tree-bootstrap 95% interval is entirely below zero; and
- tree wins exceed losses.

These criteria are frozen before prospective data but are a stricter claim
boundary, not new gates that can alter the registered experiment status.

### Dynamic-support endpoint

The paired unadjusted dynamic-depth-three versus compute-matched fixed-support
depth-three comparison must satisfy all of:

- at least 3% Brier reduction;
- its tree-bootstrap 95% interval is entirely below zero; and
- tree wins exceed losses.

Because both selectors use their original frozen risk score and differ only in
whether future support is regenerated or held fixed, this removes the
diversity-bonus selection change from the dynamic-support endpoint. It remains
an endpoint comparison rather than a claim that every internal support change
is individually beneficial.

### Truth-coverage endpoint and Brier alignment

Truth coverage is evaluated only after every root has been selected and scored.
For each candidate root, it is the fraction of the external canonical target
extensions represented in the answer-conditioned terminal support on the same
frozen policy tree. The primary Brier endpoint remains the separately frozen
held-out validation-support average; the direct association pairs these two
distinct metrics tree by tree. Canonical targets never enter support
generation, root scoring, or policy selection.

The following three paired coverage comparisons are reported separately from
the Brier comparisons:

- bonus depth three versus unadjusted dynamic depth three;
- bonus depth three versus dynamic depth two; and
- unadjusted dynamic depth three versus fixed-support depth three.

For each comparison, the candidate must have strictly higher mean coverage,
the paired tree-bootstrap 95% interval for candidate-minus-baseline coverage
must be entirely above zero, and tree wins must exceed losses. The
truth-coverage endpoint family passes only when all nine checks pass.

Parallel improvements in mean coverage and Brier do not show that coverage
explains the Brier benefit. The result therefore reports a direct association
for each of the same three policy contrasts. On each tree, coverage uplift is
candidate-minus-baseline canonical coverage and Brier benefit is
baseline-minus-candidate primary endpoint Brier, so positive values favor the
same candidate. Spearman correlation is computed only over changed-root trees;
unchanged roots create mechanical zero/zero pairs and are excluded.

The stronger truth-coverage alignment family additionally requires:

- a strictly positive changed-root Spearman correlation in each of the three
  contrasts; and
- a joint tree-bootstrap 95% interval entirely above zero for the mean of the
  three changed-root Spearman correlations.

The bootstrap resamples complete trees jointly across contrasts. Individual
contrast intervals are reported but are not gates. Association seeds are
`112810`, `112811`, and `112812`; the family seed is `112813`; all use 20,000
samples. These are simple zero-effect boundaries, not thresholds selected for
the prospective cohort.

These criteria are a stronger interpretation boundary, not registered
scientific gates. Coverage cannot change the prospective result status, rescue
a failed Brier family, alter a selected root, or authorize Block B. Conversely,
a Brier result may support the existing LLM-native dynamic claim without
establishing truth-coverage alignment.

Before this amendment, a zero-call historical mechanics replay was inspected
only to determine whether the endpoint had usable dynamic range. It was
unsaturated and directionally mixed: bonus versus unadjusted was slightly
positive, while depth three versus depth two and dynamic versus fixed were
negative. No threshold or coefficient was selected from those values; the
prospective criterion is the simple zero-effect boundary plus uncertainty and
win-direction checks above.

A second zero-call historical replay was inspected before either prospective
block solely to determine whether the association was identifiable. On the
12, 26, and 29 changed-root trees in the open fresh-32 cohort, the three
Spearman correlations were `.380`, `.172`, and `.202`, with every individual
95% interval crossing zero. Across all candidate roots, within-tree-centered
coverage versus Brier benefit was effectively uncorrelated (`rho=-.017`). This
gives a mean changed-root Spearman of `.251` with joint tree-bootstrap 95%
interval `[-.085, .507]`, which fails the prospective family criterion. The
audit motivated measuring the direct link but did not set a nonzero threshold
or authorize a causal claim.

### Ranking mechanism

The diversity-adjusted candidate ranking must have both higher mean Spearman
correlation with realized candidate Brier and lower mean candidate-set oracle
regret than the unadjusted ranking. This is diagnostic corroboration and is
reported separately from outcome gates.

## Claim Tiers

- `truth_coverage_aligned_dynamic_nonmyopic_confirmation`: every condition
  for `full_llm_native_dynamic_nonmyopic_confirmation` passes and the complete
  truth-coverage endpoint and direct alignment families pass. This permits the
  sharper claim that non-myopic selection improved truth coverage and that
  coverage uplift was prospectively aligned with primary Brier benefit across
  the frozen contrasts. It does not permit causal mediation language.
- `full_llm_native_dynamic_nonmyopic_confirmation`: registered depth and
  selector viability pass, diversity-selector superiority passes, and the
  dynamic-support endpoint passes. This permits a prospective claim that
  non-myopic planning over LLM-generated, path-dependent belief dynamics wins
  in this Number Game protocol. It does not by itself establish that improved
  truth coverage was aligned with the result. Ranking and coverage evidence are
  reported but not required.
- `nonmyopic_with_diversity_selector_gain`: registered depth and viability
  pass and selector superiority passes, but the dynamic-support endpoint does
  not. This permits a useful stochastic-diversity-feature claim, not a full
  dynamic-versus-fixed support claim.
- `nonmyopic_with_viable_diversity_selector`: the registered experiment passes
  but selector superiority is not established. This permits the prospective
  depth result and selector non-worsening only.
- `nonmyopic_depth_only`: the depth family passes but registered selector
  viability does not. Report the depth-family evidence descriptively; the
  registered combined confirmation remains a gated null.
- `mechanism_only_without_nonmyopic_depth`: selector superiority or the
  dynamic-support endpoint passes but the depth family does not. No
  non-myopic planning win may be claimed.
- `prospective_null`: none of the depth or mechanism families passes.
- `mechanics_failed`: source mechanics fail; no efficacy tier is assigned.

No tier authorizes universal benefit, monotonicity beyond depth two versus
three, coefficient tuning, retrospective relabeling, or cross-model
robustness. No tier authorizes the claim that truth coverage causally mediated
the Brier result. The coverage and alignment families cannot rescue any Brier
family. The report must independently replay the completed result, reject
non-finite or inconsistent fields, bank JSON and Markdown exactly once, and
make zero model calls.
