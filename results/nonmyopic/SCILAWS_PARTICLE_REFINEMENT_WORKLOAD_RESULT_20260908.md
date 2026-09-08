# Shared-budget refinement: no complete first root

Frozen implementation/protocol4d3b6122. Artifact SHA256
d7db4f5d7d0cd556861964cd6ded66ad3b0cfd1f48c725129a4f49170062d0c4:
SCILAWS_PARTICLE_REFINEMENT_WORKLOAD_20260908/result.json.
All three fixed fixtures terminated under the shared time guard, not a process
crash. Accepted-reference traces and terminal failures are preserved.

| History | Complete roots of8 | Accepted references including reuse | Reused | Charged evaluations | Seconds |
| --- | --- | --- | --- | --- | --- |
| zero | 0 | 110 | 8 | 38930 | 5.00008 |
| affine | 0 | 111 | 8 | 39030 | 5.00015 |
| quadratic | 0 | 109 | 8 | 38442 | 5.00004 |

All stopped during the first root action. Fourteen different child branches
were visited in each accepted prefix. The100000-evaluation cap was not reached;
wall time was the binding constraint. Small time overshoots reflect checking
between numerical operations, not a larger authorized budget. Reused work was
free, so this was already optimistic relative to cold deployment. It cannot
be presented as an almost-complete eight-root decision.

The bound/refinement logic has therefore NOT delivered a usable h2 reference.
Outer integration error was still unbounded even if terminal refinement had
finished. The result supplies neither deployment permission nor scientific
evidence against non-myopia; it rejects this numerical implementation under
the frozen runtime constraint.

## Stop And Next Requirement

Do not expand the unchanged refinement grid or run more node-count variants.
The next useful engineering step is a measured computational profile of the
full-mixture likelihood/moment integral, with a concrete throughput target
derived from complete decisions, before implementing another optimization.
A faster kernel or structurally cheaper exact inference needs independent
equivalence tests and an actual all-root budget check. An isolated h1 speedup
is insufficient, and no paid proposer calls are warranted until the numerical
policy is executable. Do not count cached reference work as a deployment gain.

This closes the current shared-budget refinement workload, not the whole goal.
The source/LLM contribution and non-myopic efficacy remain untested. Seven focused
tests passed0.80s; scoped lint passed. Source/model calls0, paid cost0; account
and London Sept8 ledger unchanged. Process exited, automation stays paused.
