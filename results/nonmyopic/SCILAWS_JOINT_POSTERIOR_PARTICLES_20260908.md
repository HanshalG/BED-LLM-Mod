# Joint posterior-particle adapter

Added explicit sample_posterior adapter from the initialized regression mixture.
For each positive posterior family, sample sigma_squared from its inverse-gamma
and beta conditional on that same variance, using the transposed Cholesky solve
required by the precision parameterization. Fixed counts per family preserve its
posterior mass through weights rather than random multinomial family allocation.
Zero-mass families remain absent; representable nonzero mass underflow fails.

Each particle supplies action means, target means and its sampled observation
sigma to the general unequal-noise QuantileGaussianModel. Conditional target-noise
variance is included only when the source model requests noisy-target risk. This
does not collapse parameter/noise uncertainty or use the equal-noise native path.

The input state is already the posterior: initial shared observations are not
replayed. New real/simulated observations use the same finite Gaussian likelihood
and normalized weight update. Finite support can lose effective sample size and
does not equal the continuous conjugate update; no rejuvenation is claimed.

Explicit RNG, bounded count1..4096 per family, conservative64MiB construction
estimate including coefficient objects/array copies; this is not measured resident
memory. Variance/family arrays are read-only and coefficients are immutable tuples.
This construction bound does NOT certify planner memory: the current batch backend
also allocates quadratic particle-distance storage and has independent limits.

Fourteen adapter/initialization tests passed in.95s. Checks cover exact seeded
correlated-precision sampling formula, family mass, varying noise, noisy/noiseless
target convention, reproducibility, immutable buffers, full subsequent likelihood
update and invalid counts. These are construction tests, not predictive accuracy
or particle-count qualification. No source data or planning audit has run.

Next prospectively freeze an all24-fixture particle-count refinement audit with
independent random streams, comparison to analytic predictive means/variances/densities,
and at least one full conditional update including family masses and effective
sample size. Use no private outcomes. Require adequate accuracy before depth runs;
do not equate arbitrary finite support with a useful LLM model space. Separate
myopic-repeat alignment and unequal-noise quadrature gates remain necessary.

No source/modelcalls/spend; authenticated usage220.376693994,balance24.623306006,
London dailyspend0. Automation paused, no active process, full goal unfinished.
