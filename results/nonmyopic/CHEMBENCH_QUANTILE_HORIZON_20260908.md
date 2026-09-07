# Quantile horizon development: bounded depth three, refinement incomplete

Added fixed-size Gaussian predictive-quantile quadrature. It finds mixture
quantiles with bounded vectorized bisection and uses full raw likelihoods for
posterior updates. The existing solver still searches all contingent actions
within the declared horizon; no greedy tail, particle truth index, or LLM call
was introduced. Integration has no intrinsic error certificate.

## Execution And Results

V1 hit the existing250,000 node cap and banked execution_failed. Inspection found
that the shared solver recomputed the already-scored root to materialize its
tree, especially after bounded-cache eviction. Reusing the scored root removes
this duplicate work without changing the objective or tie ordering. A counting
regression checks selected/nonselected root expansion. The independently
exhaustive finite-horizon tests remain intact.

V2 adds atomic partial checkpoints and applies that implementation correction;
scientific fixture settings, orders16/32, error0.001 and runtime/node caps are
unchanged. Both orders finish within the per-plan limits.

| Quantity | 16 branches | 32 branches |
| --- | --- | --- |
| Maximum one-step absolute error | 0.00530873 | 0.00085841 |
| One-step selected-action regret | 0 | 0 |
| Three-step sufficient-statistic error | 0.00016060 | 0.00003481 |
| Three-step nodes | 22865 | 173217 |
| Three-step seconds | 6.522 | 17.829 |
| Constructed adaptive risk | 0.00000033 | 0.00000181 |
| Constructed open-loop risk | 0.05712181 | 0.05310372 |

Both constructed adaptive trees measure regime first and select assay1 or assay2
after negative or positive regime observations. This is a test of contingent
decisions, not evidence of useful chemistry or LLM reasoning. The three-step
reference is an exchangeable Gaussian sufficient-statistic calculation, not
equal-real-budget monotonicity evidence.

**Terminal V2 status: synthetic_checks_failed.** The coarse one-step error
exceeds0.001 and the open-loop value changes by0.004018 between orders, also
above0.001. Positive adaptivity in this fixture cannot rescue those failed
checks. The fine-order one-step pass is not a pass for the whole instrument.

Artifacts: `chembench_quantile/20260908-v1/RESULT.json` and the V2 directory,
including source hashes, per-case errors, timings and selected continuations.

## Next Work And Scope

Focused quantile, finite/raw horizon, integration and belief regressions pass
72/72 in25.77s. Scoped lint passes. This is software verification, not a pass of
the numerical refinement gate above.

Use bounded batch evaluation to make finer quantile integration feasible,
first prove numerical equivalence to this scalar implementation, then use a
separately specified refinement audit. Retain full contingent decisions and
all controls. Do not retry these saved paths, relax the failed gates, or launch
chemistry on the basis of the positive constructed gap.

The previous goal turn was progress: it banked the nested-integration cost limit.
This turn changes code and produces the next actionable evidence; the full
three-deliverable plan remains incomplete. No paid calls, source chemistry
outcomes, cluster work or automation changes occurred.
