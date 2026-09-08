# Fixed classical reference and response model

Freeze before new source observations. Geometry remains exactly SHA
5e9f2bd902fa9de251cbe033bdd7dd4d5ad8fc79d870e80add7920ed92a18197.
This is a classical numerical opportunity reference, not an LLM result or
permission to execute source episodes. No task-specific formulas were inspected.

Map each frozen transformed input interval to [-1,1]. Four equal-mass families:
constant; intercept plus all linear terms; intercept plus linear and all quadratic
terms including interactions; intercept plus two Gaussian RBFs per axis centred
at -.5 and .5 with bandwidth .5. All coefficient prior means are zero; conditional
precision is .25 for intercept and 4 for other coefficients. Noise variance has
InverseGamma(shape=3, scale=.2). No tuning by task or by observed depth gains.
These are classical fixed features, not an exhaustive semantic model space.

Shared initial data arrive in frozen point-major replicate order. Set response
scale s=max(abs(initial responses)), or 1 if all responses are zero. Thereafter
all responses and prediction targets use z=asinh(y/s), with s never refitted after
adaptive measurements. Keep the entire transformed posterior predictive and score
future transformed observations, not an inverse-transformed mean. Initial labels
are used to set scale AND condition the model: this is explicitly empirical Bayes,
not an exact generative posterior for the full source history. Prequential claims
can begin only after that shared initialization. No hidden noise atoms/scale or
clean formula enter the prior. The Gaussian conditional residual assumption in z
is a working model requiring calibration, not the source's known noise law.

The intended terminal estimand is equally weighted squared prediction error for
the transformed future observation at the 64 fixed target inputs. This rewards
prediction of E[asinh(Y/s)|x], not recovery of the noiseless formula or raw-scale
conditional mean. All arms share the initial s and the target noise streams.
Report per-task errors and equally weighted task-level paired differences; never
pool dimensional raw units. Distributional calibration and numerical integration
accuracy remain separate gates. Full endpoint replicate counts, seeds, controls,
power criteria and source use obligations still need freezing before execution.

The immediate permitted check is public-prior runtime only, with no initial labels:
all eight tasks in saved order, quadrature order8, h1 then h2 then h3 with repeats,
5 seconds and 100000 expanded nodes per plan. Stop deeper plans for a task at the
first cap; report every task and every capped prefix. No hidden source is opened.
This is not an accuracy qualification; it uses the generic zero-mean prior and
cannot establish posterior-conditioned runtime or action rankings. Do not shrink
the eight-action menu or hypothesis pool after a cap. Optimize implementation or
bank the limit; no scientific null can be inferred from an incomplete plan.

Before any source result or proposer call, the full paired protocol must still
include receding h1/h2/h3 under the same four-measurement budget, open-loop h3,
productive-compute myopic and random controls, independent calibration, numerical
refinement and licensing review. Passing a fixed-feature reference cannot establish
the LLM is useful, nor that anticipating future LLM discovery helps. Those remain
the main scientific requirements, not optional extensions to a classical win.
