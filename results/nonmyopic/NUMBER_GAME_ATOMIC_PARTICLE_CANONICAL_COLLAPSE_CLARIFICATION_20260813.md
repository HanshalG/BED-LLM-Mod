# Atomic-Particle Canonical-Collapse Clarification

Date frozen: 2026-08-13, before implementation completion and before any model
response under the atomic-particle interface.

The parent protocol requires every positive-predictive-mass simulated branch to
produce a valid 64-particle belief. Its later canonical replay can nevertheless
reveal an answer that had zero mass under the particle belief. Such a trajectory
is genuine particle posterior collapse, not malformed endpoint data.

For all dynamic, myopic, blind, fixed, and random canonical endpoint replays,
and for each arm of the conditioned-versus-blind intermediate calibration:

- if either the retained half or generated half is empty after a realized
  canonical answer, the trajectory is marked `particle_collapse`;
- its truth-extension coverage is zero;
- its terminal posterior-predictive Brier is conservatively fixed to `1.0`;
- its intermediate canonical posterior-predictive MSE is likewise fixed to
  `1.0` and its exact-extension coverage is zero;
- no particle is injected, borrowed, repaired, or resampled from an earlier
  belief; and
- the same rule is applied symmetrically to every policy and every candidate
  root, including root-ranking fidelity.

The value `1.0` is the maximum per-coordinate binary Brier loss and therefore
cannot make a collapsed trajectory look favorable. Collapse counts and rates
must be reported for every policy and tree. All other gates, calls, seeds,
thresholds, costs, and authority remain unchanged.
