# Frozen pilot runner ready

Physics/source protocol was pushed atca1523d2 before public source predictions.
The first public preflight failed on all4 assays because all-pair integration
crossings exceeded64 nodes. No hidden worlds opened. It is preserved at
`chembench_horizon_pilot/preflight-20260908-v1/RESULT.json`.

An equal-noise numerical correction computes the upper envelope of posterior
log-density lines. It uses the envelope transitions as integration splits while
retaining all16 particles in likelihoods, forecasts and updates. Non-dominant
particles are not pruned. Unequal-noise actions retain the all-crossing rule.
This is a change to quadrature placement, not an exact error guarantee.

The new envelope32/64 synthetic audit passes all unchanged0.001 reference,
refinement and adaptivity gates, saved at
`chembench_envelope_refinement/20260908-v1/RESULT.json`. V2 public preflight then
passes all4 assays,64 branches each, with the original protocol hash and16-particle
support. Hidden worlds remain unopened in that record. No physics, seed, noise,
target or gate was changed after the preflight failure.

`chembench_horizon_pilot.py` now implements all6 deployable arms and a separately
labelled true-population h1/h2/h3 oracle diagnostic. It validates numerical/source
predecessors and caches every initial deployable policy decision before opening
the hidden-world constructor. Initial decisions are reused across worlds and
marked as reused in trajectories. The oracle particles never enter core-policy
selection. Each actual observation uses the shared world/round/design noise,
followed by the full raw-likelihood posterior and fixed-target prediction.

Each record includes actions, declared effective horizon, root action values,
observations, posterior log weights, forecasts, model risks before/after and true
target MSE. Atomic per-arm and complete-world records preserve failure prefixes.
A first incomplete world stops the panel; no shallower fallback or automatic
retry. The final gate is the previously frozen engineering screen, not a
publishability or paid-call authorization.

Focused source data/envelope/runner tests10/10 in39.71s. Synthetic tests verify
all8 worlds/6 core+3 oracle arms, paired observation indexing, no repeats, root
failure before hidden access and first-world failure before the next world.
These are mocked mechanics, not source efficacy outcomes. Scoped lint passes.

Next action after push: execute the one frozen pilot path. A runtime failure
before hidden construction remains a numerical feasibility failure; an endpoint
null must remain a null under the fixed physical protocol. No LLM calls, paid
authorization, cluster activity or automation changes are included in this run.
