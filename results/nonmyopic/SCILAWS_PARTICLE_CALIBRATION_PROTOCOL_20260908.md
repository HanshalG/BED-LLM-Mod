# Joint-particle refinement screen

Prospective: all24 declared initial fixtures, counts32/128/512 per family, seeds
1304/1305 with task/scenario-separated SeedSequence streams. Four comparisons per
draw: initialized state and independent updates at action0's exact predictive mean
plus{-2,0,2} standard deviations. These are software observations, not source data.
No particle redraw following an imagined observation.144 draws,576 comparisons.

Compare all64 target means/variances, family masses, all8 action densities at each
updated state's predictive mean+{-2,0,2}sd, and effective sample size. Per-comparison
gates: max standardized target mean error<=.05, max relative target variance error
<=.05, family total variation<=.05, max absolute log density error<=.1, ESS/N>=.1.
Every comparison/seed/case must pass at a count for full screen qualification. No
averaging away a failure or selecting a seed. Counts have separate results; no
assumption of monotone Monte Carlo error. These engineering gates are not confidence
intervals, source calibration or1e-4 planning-error qualification.

Report batch h3 workspace feasibility separately at16 branches, one belief row,
64MiB: particle-distance matrix, depth buffers and target-variance arrays. Runtime
and state budget remain untested. A count passing statistics but failing memory
cannot be used for the current planner. No source/planning/modelcalls authorized.
