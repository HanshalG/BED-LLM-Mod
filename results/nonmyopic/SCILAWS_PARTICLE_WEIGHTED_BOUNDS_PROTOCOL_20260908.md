# Analytic terminal interval feasibility

This is a new diagnostic of a bounded-search architecture. The139/144 local
accuracy screen remains failed and is not relabelled. No h2 policy run.

For the same finite Gaussian-particle posterior, every one-step action risk is
at least E[target conditional noise variance]: revealing the entire particle
leaves exactly that noise. It is at most the best linear predictor risk computed
from target/observation covariance. Use centered moments and1e-12 relative-scale
padding. These are mathematical bounds evaluated in floating point, not a
directed-rounding interval-arithmetic certificate or source-model guarantee.

For every48 fixture draw take action0 and all32 composite predictive nodes,
condition exactly, and compute8 analytic action intervals at each child.
Propagate minima using[min lower,min upper] and positive quadrature masses.
Reuse existing weighted_min_interval rather than inventing a new rule. Measure
the full terminal width and the width contributed by nodes in probability
intervals[0,1e-5] and[1-1e-5,1]. These are fixed existing rule intervals, not
failure-specific branches. Half-width <=5e-5 is only a terminal-error budget
diagnostic; outer integration error is still unknown. No root accuracy claim.

Cross-check all1152 banked independent continuation values against their analytic
intervals, including all five inaccurate approximate cases. Do not recompute the
reference integrals or rerun correction plans. All48 cases and32 children needed;
no best-case selection. Preserve all interval widths and idealized refinement
counts, which assume exact child refinements and are not measured runtime.

This work spends0 and opens no source/LLM outcomes. It changes neither scientific
gates nor budgets. A promising bound would still need a prospectively tested
refinement policy, complete decision error accounting and actual runtime before
source/depth deployment. No claim that classical bounds complete the LLM goal.
