# Number Game Truth-Coverage Power and Capacity Audit

Date completed: 2026-08-06

## Decision

**Keep the frozen Aug 8--9 diversity64 execution unchanged.**

The two fresh 32-tree blocks remain justified for the registered dynamic-support
Brier and monotonic-depth confirmation. They are not expected to make the
stronger truth-coverage family pass. The support-capacity selector is retained
only as a candidate for a separately preregistered fresh experiment.

This audit used no model calls, opened no future response or endpoint, and cost
`$0`.

## Truth-Coverage Power Boundary

The open later-fresh 32-tree cohort has the following external-canonical
coverage directions:

| Comparison | Mean coverage difference | 95% tree bootstrap | W/T/L |
|---|---:|---:|---:|
| Bonus d3 vs unadjusted d3 | +0.00473 | [-0.00852, +0.01799] | 8/20/4 |
| Bonus d3 vs dynamic d2 | -0.02273 | [-0.04830, +0.00095] | 10/9/13 |
| Dynamic d3 vs fixed d3 | -0.03125 | [-0.05966, -0.00379] | 7/8/17 |

The corresponding changed-root correlations between coverage uplift and Brier
benefit are `0.380`, `0.172`, and `0.202`. Their family mean is `0.251`, with a
joint tree-bootstrap interval of `[-0.085, 0.507]`.

A deterministic 10,000-draw plug-in simulation resampled complete open trees.
It is a retrospective design diagnostic, not a scientific gate:

| Fresh trees | Alignment-family approximate pass | Coverage-family pass | Coverage + alignment |
|---:|---:|---:|---:|
| 32 | 35.9% | 0.0% | 0.0% |
| 64 | 64.3% | 0.0% | 0.0% |
| 96 | 81.1% | 0.0% | 0.0% |

At 64 trees, the three individual approximate coverage-gate pass probabilities
are `18.1%`, `0.0%`, and `0.0%`. The registered Brier audit is much more
favorable: its pooled 64-tree primary pass probability is `90.5%`, or `69.1%`
after the conservative bonus non-worsening condition. Therefore Aug 8--9 buys
a plausible prospective Brier confirmation; truth coverage remains a stronger
interpretation-only bonus tier and cannot rescue a Brier null.

## Support-Capacity Candidate

The retrospective candidate selects the root minimizing, within each tree:

`z(predicted Brier) - 2 * z(mean future generated-support union size)`.

The coefficient was selected after inspecting both open cohorts. All results
below are exploratory and require fresh preregistration before any claim.

Across the stratified 96+32 tree pool, the capacity selector:

- beats dynamic d2 on Brier by `0.00384`, 95% CI
  `[-0.00656, -0.00124]`;
- beats fixed d3 on Brier by `0.00304`, 95% CI
  `[-0.00523, -0.00097]`;
- beats myopic on Brier by `0.01426`, 95% CI
  `[-0.01727, -0.01130]`;
- increases canonical coverage by `0.01326` versus unadjusted dynamic d3, but
  its interval reaches zero; and
- loses `0.00175` Brier to the currently frozen Jaccard selector, 95% CI
  `[+0.00007, +0.00344]`, while its `+0.00900` coverage difference is
  uncertain.

This is evidence that future generated-support capacity may be a useful
non-myopic selection feature. It is not clean enough to replace the frozen
Jaccard selector or authorize additional spend.

## Reproducibility

Both result generators were rerun to separate temporary directories and their
`RESULT.json` files were byte-identical.

- truth-coverage power result SHA-256:
  `76a5e9676818f1109210066eef57f168d243797c22977c55df98f8759f3b6b65`;
- support-capacity result SHA-256:
  `cfbb20d0e606e4dd838d8169902adc767155f1eec1e3a78459d7f34ec7661d2a`;
- model calls / cost: `0` / `$0`.
