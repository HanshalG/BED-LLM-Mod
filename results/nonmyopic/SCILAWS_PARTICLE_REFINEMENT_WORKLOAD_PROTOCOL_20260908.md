# Shared-budget all-root refinement workload

Test finite-mixture interval refinement feasibility on the fixed first task,
three histories, seed1304. This is a new bounded-refinement architecture, not
permission to deploy the failed fixed-rule continuation approximation.
Keep512 Sobol particles/family and32-node tail rule, all8 root actions and all8
possible second actions including repeats. Use existing weighted-width scheduling
and action lower bounds; no new heuristic or subset selection.

One5-second/100000-evaluation budget is shared across ALL roots per fixture,
including outer branch generation and analytic bounds. Never reset per child
or root. Refine numerical intervals until terminal half-width<=5e-5 per root.
References retain1e-7 error+tail bounds. These numerical error estimates are
not rigorous enclosures, and outer integration error remains unbounded.

Reuse the banked root0 children0/15/31 reference values rather than recompute
them. Their original work is not charged: this gives an optimistic warm-cache
workload, NOT a cold deployment runtime. A failure even with this reuse is a
useful feasibility negative; a pass would still need cold total-cost and outer
accuracy qualification. All other requested child integrals are new, and
partial action/reference prefixes must be saved. Shared reference work is
charged even when the call fails. No duplicate run on timeout.

Exclusive outputs, complete three-fixture workload, no root omission or cap
relaxation. Source/LLM calls0, no paid experiment. If no complete decision fits,
do not expand this unchanged refinement grid: determine a lower-cost operator
or tighter bound before further execution. No terminal completion of research
goal inferred from either outcome.
