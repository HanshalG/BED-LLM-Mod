# Pre-endpoint numerical correction

The first frozen pilot attempt at5bf6645d stopped in public_root_planning after
46.184s. h1/h2 root plans were saved; h2 took9.039s. h3 raised
`quantiles exceed numerical range`. completed_worlds=0 and
hidden_worlds_opened=false. The preserved run directory is
`chembench_horizon_pilot/run-20260908-v1/`.

A minimal synthetic regression reproduces the same failure with a2e-14 prior
tail mass: composite quadrature rounds an interior probability to exactly1.
The correction keeps its quadrature mass but evaluates the nearest representable
interior probability. Probabilities outside[0,1] or nonfinite values still fail.
No likelihood, particle, target, actual observation or physical noise changes.
The formerly failing test passes and the unchanged envelope refinement passes
again in `chembench_envelope_refinement/20260908-v2/`.

Before a new public planning attempt, an exact algebraic optimization also
replaces repeated terminal target tensors with the identity

    Var_p(T) = 0.5 * sum_ij p_i p_j * squared_target_distance(i,j).

Distances are precomputed once per plan, using direct differences rather than
subtraction of large moments. All target weights are preserved. The distance
matrix is charged to the existing workspace budget. No particle count, horizon,
candidate menu, branch count, approximation tolerance or resource ceiling changes.
Tests compare against high-precision weighted variance, including large offsets.

V3 of the unchanged numerical qualification passes all gates, with fine h3
runtime3.94s on the constructed case. This is not yet source h3 runtime evidence.
The preflight now requires the exact V3 source bindings. Combined numerical,
source-boundary and complete-runner tests95/95 in14.00s.

This authorizes one corrected implementation attempt on a fresh run path after
push, not a retry of unchanged code or a rescue of an endpoint null. Preserve
all prior failures. The frozen physics protocol SHA remains8e1fc9df41d177fa80b2e500c22c663e3a78a980ecedaa355c667116cc2e1d36.
Hidden worlds remain closed until every initial deployable policy completes.
No paid calls or changes to LLM-stage authorization are involved.
