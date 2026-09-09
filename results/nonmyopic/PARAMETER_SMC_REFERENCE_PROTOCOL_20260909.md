# Qualify the existing adaptive sampler against independent integrals

Do not change environments/chembench_mopen/smc.py or rerun old ChemBench cohorts.
Reuse its adaptive_tempered_smc with the previously used Sobol initialization and
full-covariance proposal, all other defaults: ESSfraction.6,3moves,80rungs,
scale.5,floor.01. Fixed512/2048particles and seeds0..7 for both fixtures.

Fixture1 is the already-opened numerical one-parameter reference, not a scientific
endpoint. Fixture2 is new: two independent uniform[-3,3] parameters, observations
a^2=b^2=1 with Gaussian sigma.1. Four equal posterior modes. Targets a,b,ab,a^2+b^2;
independent tensor-factorized Gauss-Legendre1024/2048 references must agree1e-9.

Report means, variances, evidence, mode masses, unique-particle count, moves/rungs,
likelihood rows and runtime. Do not count uniform post-resampling weights as an
independence certificate. Per-replicate qualifications: logZe<=.1 and variance
relative error<=20%; one-parameter mean error<=.01; multimodal maximum mean error
<=.15posteriorSD and every mode mass within.1 of.25. All rows retained.
Stop on600000likelihood rows or30seconds per replicate, never increase caps or
select successful seeds. No paid call, source simulation or gate reopening.

This is numerical qualification, not a new scientific effect or equal-compute
comparison with prior draws (SMC does more evaluations). A low-dimensional pass
does not qualify arbitrary generated structures. Freeze before computing SMC
comparisons; test reference and shape mechanics first.
