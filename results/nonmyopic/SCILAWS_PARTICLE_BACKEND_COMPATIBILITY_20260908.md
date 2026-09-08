# Joint posterior-particle backend compatibility

Inspected raw_belief.py, quantile_belief.py, crossing_belief.py, envelope_belief.py,
native_belief.py and batch_horizon.py at53453bd8. No source measurements or particle
experiment executed. The existing backend is reusable mathematics, not a ready
SciLaws implementation or authorization to reopen the failed ChemBench source panel.

## Three mismatches

1. Native equal-noise envelope: sampled inverse-gamma variances differ across
particles. NativeEnvelopeGaussianModel explicitly falls back to all-pair density
crossings when sigmas differ. That path may create O(P^2) crossings and fail the
branch cap. Do not plug in a posterior mean sigma to recover the fast path.

2. Target loss: both scalar GaussianParticleModel.risk and batched terminal risk
compute variance of particle target means, excluding conditional target noise.
SciLaws' frozen loss predicts a future transformed noisy observation. It needs
weighted per-particle target noise variance in addition to between-particle variance.
An adapter must preserve independent fresh observation noise, not treat a point
estimate as certain or silently change to a noiseless target estimand.

3. Repeat measurements: plan_batched removed selected actions and truncated depth
to menu size. SciLaws permits repeats with fresh conditionally independent noise.
Implemented explicit allow_repeats=False default; True retains the menu and uses
Cartesian-product open-loop sequences, without changing posterior likelihoods.
Empty menus still have depth0, horizons remain capped at3, resource controls remain.

Twenty-six focused tests pass in1.61s. New repeated-mode tests cover one/two actions,
depth1/2/3, adaptive/open-loop, full roots against scalar HorizonPlanner, and malformed
flags. Existing tests preserve defaults and quantile inversion behavior. Scoped lint
passed before the test expansion; implementation unchanged after that lint pass.

## Required adapter, not yet implemented

Draw jointly from each conditioned normal-inverse-gamma component:
sigma_squared=scale/Gamma(shape,1), beta=mean+sigma*L^(-T)*z where LL^T=precision.
Preserve posterior family mass through explicit stratified particle weights. These
are posterior particles, so shared initialization data must not be conditioned a
second time. Subsequent real and imagined raw scalar observations use the same
particle likelihood and full-weight update. Record approximation and degeneracy;
do not present finite support as exact continuous posterior conditioning.

Start with the general unequal-noise quantile path, not the equal-noise envelope.
The native inverse-CDF routine may be reusable separately, but must be checked for
the actual particle geometry and branch refinement. Preserve eight actions,64target
weights, repeat semantics, family/parameter/noise uncertainty and response scaling.

Next implement and verify conditional target-noise risk consistently in scalar and
batched paths with default zero variance unchanged. Then build the joint sampler
and compare predictive moments, likelihoods, updates and effective sample size to
the conjugate reference on declared software histories before planning. Separately
preflight state/workspace/time costs; prior backend limits and current SciLaws limits
are not interchangeable permissions. No full h3 or LLM pilot until calibration and
decision refinement are genuinely demonstrated. Source likelihood calibration and
useful LLM proposal advantage remain independent scientific requirements.

No calls/spend: authenticated usage220.376693994,balance24.623306006,dailyspend0.
No active process, automation paused. Full goal remains unfinished.
