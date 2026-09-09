# Ordering correction and independent verification

The new SingletonMLFSIterator extends the existing pinned Herb TopDownIterator.
It retains Herb's solver, constraint propagation, state snapshots, and priority
queue. Unlike the original MLFS path, it selects any unresolved hole, including
fixed-shape holes, and splits it into singleton rule choices before completion.
No unresolved uniform batch can emit a low-probability candidate ahead of the
queue. The original package installation and previous artifacts are unchanged.

For finite nonpositive rule log weights, the existing maximum completion bound
is optimistic: any missing descendants can only add nonnegative cost. With
singleton completion this supports best-first emission in descending product
weight within the declared finite grammar/depth/size constraints. This is about
search ordering, not posterior probabilities, truth coverage or optimal BED.

## Independent exhaustive checks

| Grammar | Programs | Rule assignments | Old ordering violations | Corrected violations |
|---|---:|---:|---:|---:|
| Unary, unequal weights |8|12|3|0|
| Binary, unequal weights |38|87|13|0|
| Unary/binary, unequal weights |74|204|21|0|
| Unary/binary, tied weights |74|204|0|0|

All194cases/program instances were compared with Herb's exhaustive traversal,
then independently reconstructed in Python using finite Cartesian products.
The checks require exact expression-set coverage, no duplicate emission, and
the complete expected sorted log-weight sequence, not merely local monotonicity.
They include competing fixed-shape argument combinations and probability ties.
These are small-language engineering proofs by enumeration, not exhaustive
validation of every possible constraint or the full DSL's program space.

Separate actual Julia tests verify the assignment cap throws at exactly1,
fresh iterators do not share counters, and positive log weights fail before
search. Initial integration errors were a method-dispatch ambiguity, an immutable
iterator counter, and unqualified internal completion types. They were corrected
before the successful audit; no task outcome or criterion was changed.

## Full-language check

The same353rules, saved base/guided weight vectors, known-invalid-terminal root
mask, depth5/size12 and64candidate caps were used. A hard50000rule-assignment
cap applies independently to each iterator. The full prefix log weights are
independently replayed from expression ASTs and saved weights.

| Arm | Candidates | Assignments | Ordering violations |
|---|---:|---:|---:|
| Source-weighted base |64|25228|0|
| Handcrafted guide |64|29464|0|

Enumeration took4.03/1.65seconds after loading the packages, but these are not a
fair speed comparison: the base arm ran first and includes compilation effects.
Candidate caps and maximum internal work are matched, actual work is not equal.
The production study must report both, not equate64candidates with equal compute.

Rechecking the opened handcrafted demonstration yields0base/4guided consistent
canonical programs. Guided survivors have uniformweight.25 and4distinct future
outputs. Old ordering yielded0/10; thus the correction materially changes the
finite support, while preserving ambiguity.57requests reused saved predictions
from the hash-bound preceding engineering artifact;75new isolated executions
were required. Failure predictions retain their failure status and slots.

This remains a handcrafted positive control, NOT an LLM win or a qualified
strong classical baseline on real tasks. It confirms the implementation can use
a correct structural guide and maintain multiple hypotheses. It does not establish
that the LLM can discover such a guide or that the base search budget is adequate.

## Next scientific dependency

Prepare a prospectively new Luna-medium nested-expression proposal qualification
on a disjoint source-backed cohort, using the corrected search for both guided
and source-only controls. Compare history-conditioned guidance against equal-call
history-blind guidance and an adequate symbolic search control. Keep all cases,
fixed candidate/internal-work caps, public-only feedback, and sealed predictive
targets. First require actual predictive transfer; then validate identical
real/simulated updates before any depth sweep. Do not repair/rescore closed paid
cohorts or treat this engineering result as their rescue.

26focused Python tests pass in3.44s, plus the actual Julia exhaustive/limit/full
runs. All containers exited. No API calls or spend; authenticated balance
23.609293221, conservative daily remainder4.0275717. Both previous/current goal
turns made concrete progress. Goal remains active/unachieved; no cluster or
automation changes.
