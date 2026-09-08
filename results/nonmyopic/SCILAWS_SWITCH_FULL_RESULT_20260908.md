# Shared-budget reference remains incomplete

Frozen83dbdb13 complete-root runner, one unchanged four-history execution.
Three focused tests pass in.78s and scoped E4/E7/E9/F lint passes. An injected
reference test explicitly verifies one instance/shared budget and refuses to
classify a first-root-only result as a completed plan.

Artifact SCILAWS_SWITCH_FULL_AUDIT_20260908.json SHA256:
5d7d48092e5fa4ab9271995dbc8a948ec62ee2afbec4df9521f111b6305faaf2.

All four cases reach100001 evaluations. Each has one completed root and an
incomplete second root. Every action remains null, numerical_check false;
all_complete and all_numerical_checks are false. No errors or partial roots are
silently substituted. Source/depth-three authorization remains false.

The previously completed diagnostic root was a genuine implementation gain but
not sufficient for the requested full comparison. Neither a per-root budget reset
nor pretending a one-root estimate defines an optimal policy is acceptable.
No further unchanged complete-root run is justified by these results.

## Architectural consequence for numerical work

Repeated high-accuracy inner solves are structurally expensive even after exact
algebraic speedups, coordinate conditioning and switch partitioning. Stop this
family of local adjustments as a route to deployable h3. The next candidate must
reuse information across nearby hypothetical observations, rather than repeating
the inner integral from scratch at each outer quadrature node.

Prospectively design a bounded approximation of each individual continuation
action-value function (not the already-minimized function), then take the minimum
of the approximations. This preserves the opportunity for an observation to
change the next action, while separating smooth function approximation from
switch kinks. It needs held-out pointwise comparisons against direct adaptive
inner integration, target-weighted error accounting including tails, stable
root ordering, a complete shared-budget reference check, and explicit failure
when accuracy is unresolved. It must not be treated as exact Bayes merely
because an interpolator returns a value. Freeze training/check nodes, tolerance,
tail treatment and evaluation budget before the new diagnostic; no source
outcomes or LLM calls are authorized. The existing exact solver remains the
small-case yardstick, not a solved large/deep engine.

This is still only the numerical-inference part of the plan. No LLM proposal
advantage, source-world horizon opportunity, paired monotonic depth effect,
compute-matched efficacy, or anticipated discovery has been demonstrated here.
No source observations/modelcalls, $0. Account and Sept8 London ledger unchanged,
process exited, automation paused, full goal unfinished.
