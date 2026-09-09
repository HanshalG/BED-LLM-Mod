# Saved compilation failures and prospective actionable feedback

The previous goal turn was progress: a complete paired Luna-medium qualification
changed the next action from depth testing to predictive-support diagnosis.
This turn adds evidence without paid calls, source execution or endpoint rescoring.

The static audit reads only the bound terminal manifest and24compile/repair raw
responses. It parses expressions but never executes them or reads target files.
Tests enforce this read boundary and reject a changed terminal identity.

- 9/24 batches were rejected by the closed all-or-nothing compiler.
- Of their72slots,47 are structurally invalid:23expression syntax errors and
  24computed-call arity errors.
- The other25slots are structurally valid but were discarded with those batches.
  Their semantic accuracy is unknown; no salvage or alternative score is computed.
- Four initial compile batches were invalid. Three remained invalid after repair;
  the ordinary task4 batch became structurally valid. This is observational, not
  a controlled estimate of repair effectiveness.
- The scheduled repair saw only status=invalid_batch, not compiler error type or
  location. Examples such as __bed_call1(I) recur in initial and repaired outputs.

This local interface problem is distinct from the already measured generalization
problem: even valid demo-fitting programs frequently excluded the actual query
answer. Better compilation is therefore necessary engineering, not a sufficient
explanation or a promise of a non-myopic result.

## Implemented prospective primitive

rearc_compile_feedback emits bounded per-slot error categories, locations and
generic correction hints, without echoing untrusted expressions. Invalid batches
still accept zero slots; no runtime facts, hidden outputs, extra attempts or
semantic success are invented. The example computed-call syntax is generic DSL
usage, not a task-specific solution. Valid structure is explicitly not semantics.

43focused tests pass, including six new diagnostic/feedback tests. The original
36call run replays exactly after these additions. No closed file is modified.

Next controlled intervention should isolate actionable compiler feedback from
generic invalid-batch feedback with shared initial proposals and equal repair
calls, on prospectively selected fresh tasks. Require compiler validity AND
unseen predictive support; do not replace the latter with a compilation success
metric. Freeze that design/source/cost/replay before any responses. A positive
repair result alone still cannot authorize a depth sweep: query opportunity,
coherent simulated outcomes and online/branch update fidelity remain unqualified.

Live budget unchanged: balance23.337951391, conservative London Sep9 remaining
3.75622987. No API spend, cluster or automation changes. Goal remains unachieved.
