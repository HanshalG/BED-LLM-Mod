# Predictive mixture connected to horizon planning

The previous turn completed conditional parameter/noise inference. This turn
connects fixed regression components to the existing ordinary HorizonPlanner,
without opening simulator outcomes, invoking proposals or modifying the frozen
measurement design. This is numerical mechanics, not scientific opportunity.

## Preserved uncertainty

Each component supplies its own feature matrices and conjugate prior. A state
contains all component posteriors and normalized log model weights. Conditioning
uses each component's PRE-update predictive density to update model weights,
then updates that component's coefficients and noise from the same raw value.
This is coherent conditional on a fixed model pool and prior, not a correction
for selecting new structures with the observations being scored.

Target variance is the weighted sum of within-component parameter variance and
between-component mean variance. An explicit option adds expected observation
noise for future-observation squared risk. A single surviving structure therefore
does not falsely imply zero uncertainty. Duplicating a component while splitting
its existing mass preserves forecasts, risks and branch probabilities.

Actions and targets are supplied as pre-evaluated feature matrices, with explicit
target weights. This adapter does not generate expressions, fit nonlinear shape
parameters, infer model priors or rescale responses. Those contracts remain to be
completed before a source experiment. Matrix row alignment is the caller's
responsibility; matching dimensions alone does not prove shared target identity.

## Branch integration

The predictive likelihood stays continuous Student-t. To integrate branches,
transform z = sqrt(df) u / sqrt(1-u^2); the Student-t measure becomes a normalized
Jacobi weight (1-u^2)^(df/2-1) on (-1,1). Deterministic quadrature is applied to
each mixture component and every resulting observation conditions the WHOLE
mixture. Exactly equal nodes are merged; this is not discretizing the likelihood
into observed categories or conditioning only the generating component.

Orders 2-128 and at most 16 components/8 actions are permitted. The default order
8 is a mechanics convenience, NOT a qualified scientific integration setting.
The branch count can be components times order, so an eight-action h3 run may
still be expensive. Existing planner node/time caps remain active. No full panel
runtime, heavy-tail worst case, sharp multimodal decision boundary, or h3 action
regret is certified by the tests below. Qualify these before deployment, rather
than interpreting a finite returned tree as accurate planning.

## Verification

The complete focused SciLaws suite passes: 83 tests in 5.29s; scoped lint passes.
The new six mixture tests cover variance decomposition, explicit observation
noise, split-mass duplication invariance, Bayes model-weight updates, analytic
single-component expected-risk refinement, repeated-action h1/h2/h3 integration,
invalid inputs and independent adaptive density integration of a two-component
predictive mixture. At order64 the tested one-step risk and mixture forecast
martingale error are below 1e-5. These thresholds cover the stated fixtures only.

The horizon fixture chooses the same informative action at each depth. It tests
that requested contingent depth is represented and repeat choices work. It is
deliberately NOT an example of a non-myopic win. Lower expected risk from planning
more measurements is not evidence that a deeper deployed policy wins at the same
four-measurement budget.

## Next dependency

Freeze the deployable executable-feature pool and prior/scaling/response/noise
contracts, then qualify whole-prior branch integration and bounded full-control
planning before opening source measurements. The observation estimand must avoid
the divergent exponentiated Student-t moment issue already documented. A small
numeric dictionary is only an opportunity reference; useful LLM-generated
mechanisms and real-history refresh still require fresh controlled evidence.
Source usage/attribution review remains outstanding. No paid or policy endpoint
permission is granted here, and the full research goal remains unfinished.

Authenticated credits/usage/balance: 245/220.376693994/24.623306006. Current
London ledger validates zero spend. No model calls, cluster use, automation
restart or source measurement occurred.
