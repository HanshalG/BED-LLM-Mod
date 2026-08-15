# ChemBench Adaptive-SMC V2 Protocol

Date frozen: 2026-08-15 (Europe/London)

## Motivation

V1 showed that adaptive-tempered SMC is necessary and predictively effective,
but two random 100-particle banks missed the frozen evidence-stability gate.
Static importance sampling had median ESS 1.0, and a small number of narrow-mode
outliers produced evidence differences up to 30 nats.

V2 changes only mode coverage and rejuvenation geometry. It reuses every V1
truth, observation, assay, query, likelihood, prior bound, cohort, and
scientific threshold.

## Frozen Numerical Changes

### Scrambled Sobol initialization

- Use 256 particles, a power of two.
- Generate a scrambled Sobol sequence in transformed prior coordinates.
- Use independent scramble seeds `2026083401` and `2026083402`, stably mixed
  with difficulty and domain.
- For priors with the `pKa1 < pKa2` constraint, generate a larger power-of-two
  Sobol prefix, reject invalid rows, and retain the first 256 valid rows.
- This is a randomized quasi-Monte Carlo approximation to the same conditional
  transformed-uniform prior; no bound or density changes.

### Full-covariance rejuvenation

- Preserve adaptive tempering target ESS 0.6, systematic resampling, three
  Metropolis moves per rung, and the 80-rung cap.
- After each resampling step, estimate the transformed particle covariance.
- Propose a multivariate Gaussian random walk with covariance
  `0.25 * empirical_covariance + diag((0.01 * prior_width)^2)`.
- Reject proposals outside the transformed box or pKa ordering constraint.
- The proposal is symmetric, so acceptance remains the tempered-likelihood
  ratio.

## Reused V1 Data

- 48 compound structures x easy/medium/hard.
- Source-only in-prior and already-open v4 cohorts.
- Exact V1 truth and observation seed families.
- Exact eight fixed assays and 512 query designs.
- Exact transformed log-rate likelihood.
- Correct structure only.
- No LLM, API, network call, or new endpoint.

V1 static 16/100 and SMC results are immutable references. V2 regenerates the
deterministic histories from the frozen seeds and verifies their hashes against
V1 before scoring a case.

## Frozen Gates

All conditions must pass.

1. Every regenerated prior, truth, history, and query hash matches V1.
2. Every V2 bank is finite, normalized, reproducible, and reaches temperature
   1 within 80 rungs.
3. Aggregate acceptance lies in `[0.05, 0.90]` on every cohort/tier/bank, and
   at least 95% of runs have nonzero acceptance.
4. Source-only evidence Spearman between V2 banks is at least 0.95 on every
   tier.
5. Source-only median absolute log-evidence difference is at most 1.0 nat on
   every tier. These are the exact failed V1 thresholds and are not relaxed.
6. V2 source-only aggregate mean-bank MSE is no worse than the V1 value
   `0.04656124523514992`.
7. V2 opened-v4 aggregate mean-bank MSE is no worse than the V1 value
   `0.03461759019005903`.
8. On every cohort/tier, the larger bank MSE is at most 1.5 times the smaller
   bank MSE, unless both are below `1e-6`.
9. V2 mean-bank MSE remains strictly better than V1 static 16 and static 100
   on every source-only tier and in aggregate on v4.

## Decision Rule

A pass authorizes adaptive-SMC posterior snapshots and evidence weights for the
next local scenario-tree implementation. It does not authorize an LLM efficacy
endpoint.

A failure closes V2. The next diagnosis must use its banked trajectories and
cannot narrow the prior, weaken evidence thresholds, or alter the already-open
truths.
