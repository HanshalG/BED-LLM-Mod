# Fixed integrand microprofile

No policy, independent reference integral, source observation or inference call
is rerun. Profile only pointwise arithmetic on three fixed first-task histories,
seed1304, the unchanged512/family posterior and child15 of action0. For each
of8 actions evaluate257 equally spaced points over its full +/-8-component-sigma
union domain. This is a timing grid, not a quadrature rule or accuracy experiment.

Compare current pairwise logaddexp reduction with algebraically equivalent
max-shifted exponential normalization, keeping the centered risk unchanged.
Three alternately ordered timing repetitions per fixture; record each time,
maximum pointwise discrepancy (must<=1e-10), and separate cProfile breakdown of
the old path. Keep one BLAS thread and all finite tails. No production change
or complete-decision speedup inferred from a microbenchmark.

The comparison asks whether normalization is a substantial CPU bottleneck. If
the measured improvement cannot plausibly support the full8-root requirement,
do not launch another unchanged refinement grid just to observe another timeout.
Any production optimization requires independent integral/error/work equivalence
and a new full-decision runtime gate before scientific use. Daily spend stays0;
all earlier failed numerical gates remain failed and the LLM goal unfinished.
