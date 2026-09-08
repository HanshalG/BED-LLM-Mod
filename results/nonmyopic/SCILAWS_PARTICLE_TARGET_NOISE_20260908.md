# Conditional target-noise loss

Implemented optional target_conditional_variances on GaussianParticleModel and its
quantile/envelope subclasses. A scalar or array broadcastable to(particles,targets)
is copied, validated finite/nonnegative and made read-only. Supply(particles,1) for
particle-specific scalar noise. Default zero preserves previous noiseless risk.

Risk is Var_p(target mean)+E_p(conditional target variance), weighted by the fixed
target measure. Forecasts, measurement sigmas, likelihoods and posterior updates
are unchanged. Fresh target noise is independent of query noise conditional on a
particle, including when the target input matches a previously queried input.

Applied the same loss to scalar risk, batched planning, and the batched myopic-policy
evaluator. Adaptive raw-horizon clipping now bounds risk by the target-mean range
bound plus maximum particle-weighted target-noise variance; the old mean-only bound
would incorrectly return zero when means coincide but targets remain noisy.
Workspace accounting includes the added variance arrays.

52 focused tests passed in22.14s, scoped lint passed. Tests include manual total
variance before/after observations; no likelihood/forecast change; immutable copies;
invalid variance rejection; scalar/batch full roots at depths1/2/3 with repeats in
adaptive/open-loop modes; common-noise constant shift; myopic evaluator agreement;
and independent raw integration with identical target means but positive noise.
Existing batch/repeat/raw/myopic tests pass with zero-noise defaults.

This is semantic compatibility, not continuous posterior or quadrature qualification.
With particle-dependent noise, exact expected future noise risk obeys the tower
property, but approximate integration can violate that identity. It must be checked
in numerical qualification rather than interpreted as a planning gain.

Next: joint normal-inverse-gamma posterior sampler, preserving exact family mass and
uncertainty in both coefficients and noise, then compare moments/likelihoods/updates
against the analytic reference before any depth run. Current myopic-policy evaluator
still uses distinct-action menus; its repeat semantics need alignment before a paired
SciLaws study. The unequal-noise native-envelope limitation remains unresolved.
No source observations, LLM calls, scientific claims or deployment permission.

Authenticated usage220.376693994,balance24.623306006,London dailyspend0. No active
process; automation paused. Full LLM-native non-myopic/discovery goal remains open.
