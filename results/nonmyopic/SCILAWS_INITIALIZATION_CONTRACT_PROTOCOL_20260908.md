# Initialized full-geometry contract check

All eight frozen public designs, three deterministic synthetic initial tables:
all zero; mean unit-coordinate affine signal plus replicate offsets[-.05,.05];
mean squared unit-coordinate signal plus those offsets. No seeds selected,
private states loaded, source measurements generated or planning calls made.
These labels are software fixtures, not samples from the source or prior.

Pass through the original response scaling/conditioning function and the explicit
corrected initializer. Require identical component posteriors, normalized weights,
forecast/risk, full8 actions/64 targets/4 families and6/10/14 initial observations
as appropriate. The corrected model's initial_state must be the conditioned
posterior, not a newly constructed zero-data prior. Check one subsequent update
against the old model in focused tests, leaving the old initializer unchanged.

Freeze before executing the24-case contract audit. Passing establishes only
initialization wiring. It changes no failed stress gate and authorizes no source
labels, runtime panel, accuracy claim, h3 deployment or model calls.
