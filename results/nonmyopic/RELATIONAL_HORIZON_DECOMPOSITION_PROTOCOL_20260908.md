# Retrospective horizon diagnosis, not a new efficacy experiment

Banked parent SHA86b5ad180d2f5a3e4dc6e13907a4827d2eda4cb166e88281000b7fa9f61786f3
remains `finite_reference_opportunity_null`. All four panels, no selected subset.
Reconstruct each already-opened likelihood matrix from its original seeds and
require its exact banked hash BEFORE any new reference computation. This is
replay for a new retrospective analysis, not a rerun of the failed efficacy gate.
Do not call the old pilot runner, change seeds or compute any new world response.

Use the existing multiplicity-preserving solver to obtain exact optimal B4 risk
and every root action value at horizons1..4. Shared5s/100000-state limit per panel;
reconstruction120s per panel. Bank failure prefix, no retries. For each banked
deployed policy, verify its root against the same truncated planner, then split:

    achieved_B4 - optimal_B4
      = [optimal_B4_given_chosen_root - optimal_B4]
        + [achieved_B4 - optimal_B4_given_chosen_root].

Both terms must be nonnegative exact fractions. For h3 with B4, continuation
excess must be zero because after the first query only three measurements remain.
Preserve root vectors to support the explanation; no formulas/labels are emitted.
The first term is root regret, the second continuation excess, not estimated
LLM error. No horizon definition, endpoint, gate, paper tier or paid permission
changes. Even a large h4 gain is post-hoc diagnostic evidence, not a rescued study.
Freeze this diagnostic and its synthetic identity tests before computing it.
