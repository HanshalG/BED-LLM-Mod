# Fixed-grid value interpolation fails validation

Frozen7bdf4e8a before one four-history/shared-budget execution. Three focused
interpolator/gate tests pass in.74s; preceding combined interpolation/tail tests
7/7 in.97s. Scoped E4/E7/E9/F lint passes. No production solver or source model
is replaced by this diagnostic.

Artifact SCILAWS_VALUE_SURROGATE_AUDIT_20260908.json SHA256:
77a0c95b634c8b9c074a22cb504a6e45028767e1ec3ded66cca83fbc5ce77861.

| History | Evaluations | Root0 check error | Root1 check error | Status |
| --- | ---: | ---: | ---: | --- |
| Empty | 100001 | .00382337 | Incomplete | Budget cap |
| Positive | 88106 | .00031630 | .00028871 | Failed |
| Negative | 91286 | .00029755 | .00024809 | Failed |
| Contradictory | 81506 | .00118096 | .00087999 | Failed |

The fixed normalized max-error gate is2e-5. All seven completed root fits fail;
no interior planning score is emitted. Normalized reported inner error is below
8.81e-9 throughout, while the interpolation discrepancies are orders of magnitude
larger. Thus the failure is not explained by the reported direct-integral error.
Tail bounds passed but do not rescue inaccurate interior interpolation.

The65-node asinh/PCHIP candidate is not qualified. Do not deploy it, relax its
threshold, or rerun with a flattering history subset. There is no full-plan
reference or source-world positive result here.

## Next decision

Uniform placement over a very wide required interval allocates work without
regard to local curvature. A genuinely new candidate would allocate nodes using
local interpolation error under the same total budget, retaining the two action
functions separately. Start with a frozen small grid, refine intervals failing
the unchanged check, reuse exact previously computed values, and validate on
fresh points disjoint from all fitting/adaptation points after convergence.
Do not label reused adaptation points as held-out tests, or sampled error as a
uniform proof. The fixed-grid route remains failed; a new adaptive algorithm
needs its own prospective specification and adversarial tests before execution.
If it also fails the budget/accuracy conjunction, more interpolation polish
alone cannot justify progression to LLM calls.

No source observations/modelcalls, $0. Account and Sept8 London ledger unchanged.
Process exited, automation paused, full scientific goal still unfinished.
