# Bound-ordered adaptive search

Implementation and frozen audit pushed at bb7f43c6. Adds an explicit
use_action_bounds option, default false, for adaptive search only. The model
provides a forced-first-action family-revelation lower bound for the corrected
numerical objective. The full action menu with repeats in the potential is a
relaxation even when the actual policy has a restricted or nonrepeat menu.

Actions are ordered by their bound and evaluated until a bound strictly exceeds
the feasible incumbent by a conservative 1e-10 relative/absolute margin. Ties
are not pruned. Evaluated violations, nonfinite bounds and unsupported modes
fail closed. Provider validity remains essential: a falsely high pruned bound
cannot be detected merely by checking evaluated actions. This hook is not
enabled on the old uncorrected model.

Returned exact root_action_values and root_pruned_lower_bounds are disjoint;
their union covers all root actions. Pruned bounds are never presented as exact
risks. pruned_actions includes work encountered during policy materialization.
All evaluated batched leaves retain their original node charge.

## Verification

198 focused SciLaws/ordinary/raw/batched planner tests pass in 22.11 seconds.
New tests compare full contingent trees, values, chosen actions, evaluated root
values and root-bound coverage against exhaustive search at depths 1/2/3, with
single/mixed families and repeats allowed/forbidden. They independently compare
model bounds with family-oracle enumeration and test restricted menus, no-prune
ties, invalid bounds and evaluated violations. Scoped E4/E7/E9/F lint passes.
Initial test run had a missing required target_weights fixture argument; fixed
before the successful full run. Unrestricted lint also reports broader style
rules; no claim of repository-wide lint cleanliness is made.

## Frozen public-prior audit

Artifact SCILAWS_BOUND_SEARCH_AUDIT_20260908.json SHA256:
b308849c7bce5d89b984c550fd2df5aed6a7fd0c1a4fa21ff1013e8fc9d6c98f.

Eight unchanged tasks, eight actions, order 8 quadrature, five seconds and
100000 counted nodes per plan. Executed once, sequentially, from pushed code.

- Depth 1: all eight complete, seven root actions pruned in each case.
- Depth 2: all eight complete in 0.627-0.738 seconds, 3169-13249 nodes.
  Seven cases with a completed preceding exhaustive horizon-corrected audit
  have exactly equal saved root action and risk (zero absolute difference).
  Baseball now completes but has no preceding completed exhaustive value.
- Depth 3: all eight hit max_nodes. No completed depth-3 value or policy is
  reported. This is not a full runtime pass.

The strict numerical pruning is useful, but not enough. Same-dimensional public
priors remain identical by construction; these are not independent source-world
results, nor evidence of useful LLM discovery or horizon efficacy.

## Next dependency

Investigate partial chance-branch bounds: retain the fixed analytic correction,
combine exact contributions of evaluated branches with valid continuation lower
bounds of all remaining branches, and abandon an action only once this total
exceeds the incumbent. Such an abandoned action must remain explicitly bounded,
never cached or returned as an exact value. Preserve node accounting, ties,
full menu, full likelihood, caps, and exhaustive equivalence tests. This would
be a new search implementation, not authorization to rerun unchanged failures.

Mixed-family integration accuracy, source usage obligations, sealed source
measurement protocol, LLM proposal calibration and paired sequential endpoints
remain unresolved. Search equivalence does not establish those claims.
No source measurements or model calls; $0 spend. Authenticated balance
24.623306006 and Sept8 London ledger unchanged. Audit process exited, automation
paused, full research goal unfinished.
