# Parameter and noise inference component

The previous geometry turn was progress. This turn implements and verifies the
missing unknown-noise regression primitive. No source measurements, hidden laws,
holdout states or model responses were opened. This is not an opportunity result.

## Model and implementation

For a fixed supplied feature vector phi(x), use

    y | beta, sigma2 ~ Normal(phi(x)' beta, sigma2)
    beta | sigma2 ~ Normal(m, sigma2 Lambda^{-1})
    sigma2 ~ InverseGamma(a, b)

The implementation requires a > 1 so squared prediction risk exists. It maintains
coefficient uncertainty and noise uncertainty together, rather than plugging in
the simulator noise or an estimated coefficient vector. A rank-one conditioning
step returns a new immutable, hashable state and the predictive log density
computed BEFORE the update. No hidden-world interface exists in this component.

The standard conjugate posterior and Student-t predictive formulas follow the
normal-gamma linear-model calculation in
[Christen, Bayesian Linear Models (2024), sections 2-4](https://arxiv.org/html/2406.01819v1).
The variance parameterization here is the inverse of that paper's precision.
These are established inference formulas, not a methodological novelty.

The component can condition on nonlinear feature functions supplied by an
executable mechanism, but coefficients enter linearly. Nonlinear unknown shape
parameters are NOT integrated by this module. Nor does it implement structure
generation, structure selection, mixture weights or multi-step integration.
It can be an analytically integrated calibration layer, not a replacement for
the LLM's open-ended executable mechanism proposals.

## What this does not establish

Conditional observation noise is iid Gaussian with shared unknown variance.
Student-t marginal predictions arise from uncertainty in those parameters;
they do not make the conditional residual process a robust, independently
heavy-tailed, heteroscedastic model. SciLaws uses local empirical residual noise.
Direct prequential predictive checks remain necessary. A failed calibration
cannot be repaired by describing conjugacy as correct for the source world.

Latent target variance and future-observation variance are exposed separately.
The latter includes expected noise variance. Under a log-response model these
moments are in log space. Exponentiating a Student-t variable generally has no
finite positive exponential moment; exp(predicted log mean) is not the Bayes
action for raw squared error. The terminal estimand must be settled before
measurements: log-space error, or a different distribution with appropriate
raw-scale moments. No response transform or endpoint is silently fixed here.

Accumulated predictive log densities are marginal evidence only for a model and
prior fixed independently of the scored observations. If an LLM selects a
feature structure using those same observations, replaying evidence does not
correct that selection. Real-history refresh needs the separately specified
prequential or validation weighting rule; artificial zero mass for new models
and duplicate-count prior inflation remain prohibited.

## Verification and next action

Eight tests pass in 1.64s. They compare scalar densities with scipy.stats.t,
sequential posterior and accumulated evidence with independent batch formulas,
order invariance, state immutability/hashability, latent/noisy variance identity,
noise growth after a surprising observation and invalid-prior/input refusal.
Lint passes. This does not qualify horizon quadrature or source calibration.

Next integrate fixed executable model components into a predictive mixture with
within-model parameter variance retained. Freeze actual feature/parameter priors,
response scaling, residual treatment, terminal estimand and controls; qualify
planning integration on synthetic predictive distributions before source runs.
The full eight-task scientific protocol and source licensing obligations remain
unfinished, so neither measurements nor paid inference are authorized.

Authenticated account credits/usage/balance remain
245/220.376693994/24.623306006. The London ledger validates $0 spend today.
Automation remains paused; the strongest LLM-native non-myopic goal is active.
