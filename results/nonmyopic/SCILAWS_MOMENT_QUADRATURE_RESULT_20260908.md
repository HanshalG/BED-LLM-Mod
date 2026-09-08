# Moment-conserving integration candidate

The previous family-bound audit exposed optimistic risk integration. This turn
implements an explicit alternative subclass; default Jacobi calculations and
their banked results remain unchanged. No source measurements were generated.

## Numerical rule

Keep every raw observation node and continuous component likelihood. Adjust
nonnegative integration masses to minimize L1 departure from the old masses,
subject to conservation of E[r_k(Y) Z^j] for j=0,1,2 and each component k.
Here r_k is the exact posterior responsibility and Z is the observation centred
and scaled by the mixture predictive mean/standard deviation. Right-hand sides
are the analytic prior component mass times its predictive moments in Z.

These identities conserve model mass, expected coefficient means and expected
noise variance after conjugate conditioning. Zero integration weights may remove
observation nodes, not latent hypotheses. Scalar branches and the batched terminal
hook use the same corrected masses, so displayed tree and objective remain
consistent. No negative weights, likelihood clipping, noise reduction or fallback
to the original rule is allowed on failure.

The bounded linear program uses [SciPy 1.15.3 HiGHS](https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.linprog-highs.html):
one-second solver cap,10000 iterations,1e-9 feasibility settings. Normalized
weights must satisfy all original moment constraints within5e-9; otherwise stop.
This is floating-point numerical validation, not interval-certified integration.

## Tests and frozen audit

Five new tests cover component probability/mean/noise conservation, identical
likelihood updates, infeasible-support refusal, analytic single-component risk
and independent adaptive density integration at order64 (error<1e-5).
109 focused SciLaws tests pass in5.48s; lint passes.

The audit was committed/pushed at33baf6f6 before execution. It covers every task
at the public prior and the same artificial history (action3,value1.7) already
used in mechanics tests. Orders8 and16 are compared on all eight actions against
the old order128 reference. Each row requires maximum absolute action-value
error<=1e-4 AND selected-action reference regret<=1e-4. No task substitutions.
The high-order reference is not a rigorous exact integral or source ground truth.

Artifact SHA256:
27006bfb58c8cea45bbbbb0f1369388004e8dd13b9c3f6e713cc2aa7e28707d3.

| Rule | Complete passing rows | Largest error among evaluated rows | Outcome |
|---|---:|---:|---|
| Corrected order8 | 2/16 | 1.77e-8 | 14 solver-status4 failures |
| Corrected order16 | 16/16 | 1.86e-8 | All tested rows pass |

Evaluated rows all select a zero-regret action under the reference. The failed
order8 rows have no complete action vector; their error must not be treated as
zero. Status4 indicates a solver problem, not proof that the positive moment
constraints are infeasible. The combined audit is false, not an overall pass.
Same-dimensional cases share normalized geometry; counts are not independent
empirical evidence. The apparent accuracy at16 does not authorize choosing a
passing subset or claiming the full rule is qualified.

## Decision

Do not launch another h3 or source episode yet. Order16 increases branching and
does not solve the existing100000-node constraint. Investigate the constraint
system's numerical conditioning at8 while preserving validation against EVERY
original constraint. A rank-aware solve may be possible, but discarding a
difficult constraint or loosening tolerances would not be the same correction.
Then test multi-step value refinement and bounded runtime; a one-step moment
identity alone cannot certify nonlinear terminal utility or future policy choices.
No pruning authorization is granted by either order.

No model calls or paid cost; live account/ledger unchanged. No processes remain
active. Automation remains paused. Source calibration, full paired protocol,
licensing review and LLM-native discovery evidence remain unfinished.
