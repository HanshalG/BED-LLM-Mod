# ChemBench Adaptive-SMC V3 Protocol

Date frozen: 2026-08-15 (Europe/London)

## Purpose

Resolve V2's sole failure: posterior-predictive disagreement between two
otherwise evidence-stable 256-particle banks. V3 doubles each bank to 512
particles and changes nothing else.

## Frozen Configuration

- Scrambled digitally shifted Sobol initialization.
- 512 particles per bank.
- Regularized full-covariance random-walk rejuvenation.
- ESS target 0.6, three moves per rung, maximum 80 rungs.
- New seed bases `2026083501` and `2026083502`.
- Exact V1/V2 prior, truth, history, query, likelihood, cohort, and structure.
- Correct structure only.
- Every case must replay the immutable V1 hashes.
- No LLM, API, network call, or new scientific endpoint.

## Source-First Ordering

Run the 144 source-only cases first. The already-open v4 replay may run only if
all source gates pass. This prevents spending local compute on a configuration
that has already failed its primary dependency.

## Source Gates

1. All V1 hashes match and every run is finite, reproducible, reaches
   temperature 1 within 80 rungs, and has healthy acceptance in `[0.05, 0.90]`.
2. Evidence Spearman is at least 0.95 and median absolute evidence difference
   is at most 1.0 nat on every tier.
3. Mean-bank source aggregate MSE is no worse than V2
   `0.04252595039735338`.
4. Mean-bank MSE remains better than static 16 and static 100 on every tier.
5. For every tier, the larger bank MSE is at most 1.5 times the smaller bank
   MSE, unless both are below `1e-6`. This is the exact V2 failure threshold.

## Opened-v4 Gates

If and only if source passes:

6. All hashes and health conditions pass on v4.
7. Aggregate mean-bank v4 MSE is no worse than V2
   `0.016629854723866702` and remains better than both static baselines.
8. The 1.5 bank-MSE ratio passes on every v4 tier.

## Decision

A full pass authorizes the 512-particle posterior representation for local
scenario-tree development. It does not authorize an LLM efficacy endpoint.

If source fails, stop without v4. If v4 alone fails, bank the stress failure and
keep posterior integration in source-only development. Do not increase beyond
512 particles in this line. The next repair must use experimental design to
target posterior-predictive disagreement or explicitly preserve a multi-bank
mixture.
