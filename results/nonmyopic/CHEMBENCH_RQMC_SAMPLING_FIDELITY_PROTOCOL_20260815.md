# ChemBench Randomized-QMC Sampling Fidelity Protocol

Date frozen: 2026-08-15 (Europe/London)

## Purpose

Test whether randomized quasi-Monte Carlo makes posterior-sampled one-step
action values faithful at 256 trajectories per action. The predecessor IID
gate failed full-action ranking while passing practical regret gates.

This is a prospective variance-reduction successor. It changes no posterior,
truth, history, action, target, reference value, metric, or threshold.

## Frozen Bindings

- IID sampling result SHA-256:
  `d71964ff247bc80408b0b5c78c4b168c5117ece372446990c4a5dfd17b42c0f6`.
- Pooled reference result SHA-256:
  `637c031946268e8015687d5eb4af4d037bfc1399300e56c0e50bdf049b4fab07`.
- Component-bank reference SHA-256:
  `e4533d76af6ab9be344bbf72806762eb695a425e7664a272105b3d075f706b16`.
- Authorized V3 result SHA-256:
  `dce0832190aa8b76c9b345220a9043edb6e622db3b8cf0f524b68b5c06c783c1`.

Use the exact predecessor 36 cases, pooled 1,024 particles, 14 actions, 128
task targets, likelihood, saved 2,048-outcome references, and component-bank
references. Regenerated pooled particle hashes must match before scoring.

## Randomized QMC Trajectories

For each case and replicate, generate a digitally shifted two-dimensional
Sobol prefix of length 256 using the repository's bound Sobol implementation.
Use seed base `2026083900`, stably mixed with replicate, difficulty, and domain.

- Coordinate one selects a pooled posterior particle through its cumulative
  weight distribution.
- Coordinate two maps through the standard-normal inverse CDF and supplies
  transformed observation noise.
- Clip the normal coordinate only to `[2^-52, 1-2^-52]` before inversion.

Use the same ordered `(particle, noise)` pairs for every candidate action in a
case and replicate. This common-random-number coupling estimates action-value
differences directly. Do not mix the action identifier into the Sobol seed.

Evaluate nested prefixes `32`, `64`, `128`, and `256`. Use four independent
digital shifts. The 1,024-trajectory four-scramble mean is diagnostic only.

## Metrics And Gates

Use the exact predecessor metrics and gates. The 256 budget opens oracle-depth
development only if all four randomized replicates independently have:

1. finite and reproducible bindings, particles, coordinates, outcomes,
   posterior weights, and risks;
2. median within-case Spearman at least 0.90;
3. at least 90% of cases with Spearman at least 0.80;
4. at least 90% of cases with pooled normalized top-one regret at most 3%;
5. mean pooled normalized top-one regret at most 1%;
6. under each component-bank reference, at least 90% regret coverage at 3%
   and mean normalized regret at most 1%.

Report prefix metrics, pairwise selected-action agreement, and the four-shift
ensemble. Lower counts and the ensemble are non-gating.

## Decision

A pass freezes common-random-number RQMC with 256 trajectories per action for
the oracle-support horizon-opportunity gate. It authorizes no LLM call.

A failure closes the full 14-action sampled planner. The next admissible move
is a prospectively defined action-shortlist gate or a different environment
with a stronger numerical horizon gap, not additional samples, relaxed rank
coverage, or LLM spend.

This protocol uses no planning depth, structural proposal transition, LLM,
API, network call, benchmark endpoint, or paid resource.
