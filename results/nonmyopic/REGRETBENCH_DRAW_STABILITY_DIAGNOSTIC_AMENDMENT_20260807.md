# RegretBench Draw-Stability Diagnostic Amendment

Date: 2026-08-07

Status: frozen before any RegretBench policy response or endpoint.

## Motivation

Dynamic depth two averages two answer-conditioned LLM support draws for every
root and simulated truth. Common random numbers remove root-specific seed
luck, but two draws can still yield an unstable argmin. A null result should
distinguish an incorrect non-myopic score from a noisy root selection without
creating a post-outcome subgroup or rescue rule.

## Exact Diagnostic

Using only the already frozen conditioned branch responses, independently
compute terminal truth-group Brier and log-loss root risks for draw zero and
draw one separately. For every task, serialize:

- both `4`-root draw-specific risk vectors;
- each draw's minimum-Brier root with lowest-index tie breaking;
- whether the two draw-selected roots agree;
- whether both draw-selected roots equal the original two-draw averaged
  dynamic root; and
- the averaged dynamic winner's Brier margin over the second-best averaged
  root.

Aggregate across all 64 tasks:

- counts and fractions for draw agreement and both-draw agreement with the
  averaged selection;
- mean, median, minimum, and maximum averaged winner margin; and
- descriptive dynamic-minus-`myopic_refresh_brier` realized Brier means on
  stable and unstable tasks, with task counts. A task is `stable` exactly when
  its two draw-selected roots agree; all other tasks are `unstable`.

## Boundary

This diagnostic uses zero new model calls, prompts, responses, seeds, paths,
or endpoint accesses. It cannot alter root selection, mechanics, science
gates, development or confirmation status, confirmation authorization,
claim tier, or any preregistered full-cohort comparison. Stable and unstable
subsets are descriptive only and cannot rescue a null full-cohort result.

The producer and independent verifier must recompute the diagnostic exactly
from raw initial and branch supports. The frozen report and paper fragment
must label it as non-gating and non-rescuing.
