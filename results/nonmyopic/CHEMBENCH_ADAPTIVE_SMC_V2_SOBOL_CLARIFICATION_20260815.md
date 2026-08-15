# ChemBench Adaptive-SMC V2 Sobol Clarification

Date: 2026-08-15 (Europe/London)

This implementation clarification is frozen before any V2 calibration particle
or response is generated.

The repository environment contains a broken `scipy.stats` import because its
array-API compatibility probe encounters the repository's lightweight `torch`
module. V2 therefore uses an internal standard Sobol direction-number table for
the maximum eight ChemBench parameter dimensions and applies a seeded digital
XOR shift to every coordinate. The resulting sequence remains a scrambled
Sobol prefix; this changes no V2 scientific choice.

The first eight primitive polynomials and direction seeds are fixed in source,
and unit tests require the generator to be deterministic, ordered-support
valid, and marginally well spread. No V2 seed, particle count, prior, proposal,
history, outcome, threshold, or gate changes. No V2 particle, model call,
network call, endpoint, or cost occurred before this clarification.
