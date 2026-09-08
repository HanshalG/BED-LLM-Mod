# Streamed overlap bound: cost/pruning screen

Implement the Gaussian-overlap lower bound derived in the kernel-profile result.
It retains full finite support, unequal noise, noisy targets and exact likelihood
semantics. Sum pair contributions in blocks under64MiB conservative temporary
workspace allowance, never materialize all particle-pair-target differences.
Count particle-pair terms separately from observation evaluations; neither is
free or a model call. Shared5s cap for an8-action interval menu.

Before any full grid, use the first task's zero/affine/quadratic fixtures,
seed1304, action0 child15 at fixed32-node tail rule and512 particles/family.
Reuse the24 independent reference values already banked for these states.
No old reference integrals, approximate plans or source responses rerun.

Workload gate requires, in EACH of these three states: all reference values
contained within interval bounds allowing their saved numerical errors; at
least one action eliminated by lower>best upper; child-minimum width at most
half the prior noise-only interval width; total overlap-bound time <=0.5s.
These are prospective practical usefulness criteria, not a scientific efficacy
threshold. Runtime cap5s is a guard;0.5s gate is already lenient for many-child
use and a pass would still require amortized full-decision cost evaluation.

If any criterion fails, bank the result and do not expand the unchanged overlap
grid. Mathematical validity alone is insufficient. Four tests cover independent
integral containment, identical-likelihood equality, block-size equivalence and
workspace rejection. Ordinary floating-point padding is not directed rounding.
No previous gate changed and no source/LLM/deep search authorized.
