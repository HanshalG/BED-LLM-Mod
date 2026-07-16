# COPEx Early Audit for Strategy-Prior L3

Date: 2026-07-16. This is the bounded early audit required by the StrategyEIG
revival protocol. It does not launch or integrate a new environment.

## Sources

- Paper and official code link: [OpenReview, *Constrained Bayesian Experimental
  Design via Online Planning*](https://openreview.net/forum?id=47PywA0l3h).
- Public repository: [yujiag21/COPEx](https://github.com/yujiag21/COPEx), audited at
  commit `e19e2e11aa19098f09abbc3336b5c442bf88cb24`.
- Paper identifier: [arXiv:2605.26990](https://arxiv.org/abs/2605.26990).

## Released Location Task

The repository contains a self-contained `Location_budgeted` task and configuration:

- one latent source `theta` with a uniform prior on `[0, 1]^2`;
- designs `xi` in `[0, 1]^2`;
- mean observation
  `log(0.1 + (1e-4 + ||xi - theta||_2^2)^-1)`;
- additive Gaussian observation noise with standard deviation `0.5`;
- 30 sequential designs;
- a hard transition constraint `||xi_t - xi_(t-1)||_infinity <= 0.1`;
- one initial design sampled uniformly under the trial seed.

This is in-contract for the current goal: the source is a ground-truth latent target,
the likelihood is analytic, and the transition constraint creates state coupling that
makes future design availability depend on the current action.

## Reproducibility Audit

The task equations and constraint code are present. The end-to-end COPEx runner is not
currently reproducible from the public repository alone:

- its configuration requires
  `ALINE/models/location_finding/inference/inference_model.pth` and
  `ALINE/models/location_finding/design/design_model.pth`;
- neither checkpoint is present in the repository or its single published tag;
- the runner raises `FileNotFoundError` when either checkpoint is absent;
- the repository has no license file at the audited commit.

Therefore the released COPEx planner is not a cheap executable reference arm. Treating
it as one would require obtaining checkpoints and licensing clarification from the
authors, which is outside this bounded audit.

## L3 Decision

Adopt the published COPEx `Location_budgeted` **task specification** for L3 by
independently implementing the equations above in this repository's exact-likelihood
particle machinery. Do not copy or vendor COPEx implementation code. The L3 controls
remain those registered in the goal: StrategyEIG, random strategies, shared d1,
matched width, and discretized-grid receding d2 with a resolution/cost sweep.

This gives L3 external task provenance at low integration cost while keeping the
measurement exact. Any final report must state explicitly that it evaluates our
controlled strategy-prior decomposition on the COPEx-defined task; it does not claim
an empirical comparison against the unreproducible released COPEx package.
