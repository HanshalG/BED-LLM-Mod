# Joint action integration: accurate, insufficient whole-decision throughput

Frozen code/protocol67bfe5f7. Artifact SHA256
318f9ab4c2e1cf0b81341b93ae5fec1adb1b92e58b3a7ba4ed27fdd3574ed4aa:
SCILAWS_BATCHED_REFERENCE_WORKLOAD_20260908.json.
Allthree eight-action integrals completed,24 banked values matched; process
exited normally. No scalar reference integrals or policy runs repeated.

| History | Seconds/menu | Charged posterior evaluations | Max reference discrepancy |
| --- | --- | --- | --- |
| zero | 0.07167 | 1848 | 1.74e-17 |
| affine | 0.09455 | 2520 | 1.67e-16 |
| quadratic | 0.06492 | 1848 | 8.33e-17 |

All accuracy checks passed, including action-specific mass and numerical error
checks. Allthree fail both the0.04s and800-evaluation practical targets. There
are231/315/231 callbacks, but each performs8 posterior evaluations: callbacks
must not replace the actual work count. The start-with-one-interval adaptation
and matrix operations reduce cost, but the achieved throughput is not sufficient
for the proposed full decision. No mesh retuning or repeated timing is performed.

Even the optimistic120-child-menu scenario used to set the targets would imply
221760-302400 posterior evaluations at these measured per-menu counts, above
100000, and roughly7.8-11.3s before overhead if menu times were representative.
Those are extrapolations, not an executed full tree; other child states could
be easier or harder. They explain why CPU acceleration alone cannot qualify
the current reference-refinement scheme.

## Next Constraint

Do not run another unchanged depth/refinement grid. The next architecture needs
fewer required integrations or principled allocation of precision across the
entire expected-risk calculation, not only faster arithmetic. Any such design
must account explicitly for the resulting root error and unresolved branches
and preserve the final1e-4 decision-value requirement. Previous local/reference
gates remain as recorded; do not relabel them or silently loosen tolerances to
make this workload pass. A new error-propagation algorithm requires independent
mathematical tests and a prospective complete-decision protocol before use.

The joint integrator is retained as an optional tested component, not installed
as a deployment default. Its narrow-mode test fails closed; numerical quadrature
error estimates are not universal guarantees. Four focused tests passed0.67s,
scoped lint passed. Source/model calls0, paid cost0, account and London Sept8
ledger unchanged. Automation stays paused; the LLM-native non-myopic scientific
goal and source/LLM efficacy stages remain unfinished.
