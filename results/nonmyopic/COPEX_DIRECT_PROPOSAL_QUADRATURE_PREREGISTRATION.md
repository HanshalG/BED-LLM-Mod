# COPEx Direct-Proposal Quadrature Pilot Preregistration

Frozen on 2026-07-18 after the completed Monte Carlo direct-proposal pilot and before
this runner executes any live quadrature-policy outcome.

## Evidence Being Addressed

The completed boundary-amended pilot is mechanically valid but not promotable:
LLM d2 minus shared LLM d1 entropy-AUC was `-0.0402` nats with 95% paired bootstrap
CI `[-0.1761, +0.1119]`; d2 minus call-matched width was `+0.1506` with CI
`[-0.1153, +0.4401]`. It used only four stochastic outer branches and eight stochastic
child one-step rollout samples, so both the d1 root ordering and the d2 child-action
ranking retained avoidable simulation variance.

## Change

The LLM angle-cell interface, direct continuous task, boundary projection, particles,
posterior, root-sharing rule, width control, grid arms, seed structure, and endpoints
are unchanged. Only the numerical score estimator changes:

- Every immediate candidate EIG, including every d1 and d2 child candidate, is now
  evaluated over the **entire finite particle posterior** with three-node Gaussian
  quadrature for the standard-normal observation noise (`z = -sqrt(3), 0, sqrt(3)`;
  weights `1/6, 2/3, 1/6`).
- d2 continuation proposals still require sampled child beliefs. This pilot uses
  eight source strata and eight equally weighted normal-quantile strata, randomly
  paired once per CRN decision state. Thus each root scores eight child belief states
  rather than four, while preserving common random numbers across its candidates.
- The width arm receives exactly `1 + K_realized * 8` current-state proposal cells at
  each nonterminal decision, matching d2's logical child-cell allocation.

This is a low-variance numerical-rung test, not a new proposal or environment search.

## Pilot

- Fresh seed `46022`; 8 paired trials; 8 queries; 48 prior particles plus truth.
- Three LLM direction angles per cell; generator remains non-thinking Gemma 26B,
  temperature 0, output cap 512, and adapter concurrency cap 128.
- Trial concurrency 8; one validation repair; no-padding boundary projection.
- `$2.50` run cap. The expected request volume is roughly twice the successful
  Monte Carlo pilot and remains far below both run and project budget.

## Gate

The primary endpoint remains paired entropy-AUC reduction. This pilot is promotable
only if LLM d2 has positive mean entropy-AUC gain against both shared LLM d1 and
call-matched LLM width, all legality/root-sharing/width mechanics pass, and the
true-particle log-posterior AUC contrast against d1 is non-negative. Promotion means
a 30-trial fresh-seed confirmation using this exact quadrature configuration.
