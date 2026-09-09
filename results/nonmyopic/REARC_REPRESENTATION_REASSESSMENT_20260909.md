# Representation reassessment after paired repair

Previous goal turn was progress: an actual paired repair null separated structural
rescue from sufficient predictive support. This turn inspects only banked public
feedback and pinned DSL definitions; no new model, generator, candidate execution
or withheld label was opened. The next intervention should change representation,
not simply add a DSL repair round.

## New saved-data evidence

Across six shared compilations, three batches were invalid. The other24slots
contained8visible matches and16mismatches. Actionable repairs had one invalid
batch; their40accepted slots contained22matches,10execution errors,8mismatches.
Generic repairs had three invalid batches;24accepted slots contained21matches
and3mismatches. Slots are dependent and duplicated; these counts are diagnostic,
not task-level effect estimates or generalization scores.

On06df4c85, a repaired expression includes compose(matcher(color,SIX),color).
Pinned source e5b7f1d defines matcher(f,t)(x)=f(x)==t and color(object) as the
color in a colored-coordinate object. The composition applies color twice,
passing an integer into an object operation. All eight actionable repairs fail
with TypeError at extract/mfilter. Syntax rescue did not repair the type flow.

On7b6016b9, the verbal plan says to preserve1-valued wall cells and distinguish
connected zero-valued regions. The implementation computes components after
replace(I,ZERO,ONE), erasing exactly the wall/background distinction it needs.
Saved feedback reports139wrongcells and an original wall cell changed to3.
This demonstrates plan/implementation inconsistency on the visible example;
it does not establish that the verbal rule is correct on withheld examples.

## Primary-source cross-check

[Poetiq solver](https://github.com/poetiq-ai/poetiq-arc-agi-solver/blob/a6947cff5f94244340309d4fc240f9aef55ff3a7/arc_agi/solve_coding.py)
generates executable code and incorporates execution feedback. It returns once
all training examples succeed. That stopping rule does not preserve a predictive
program posterior. Its [sandbox](https://github.com/poetiq-ai/poetiq-arc-agi-solver/blob/a6947cff5f94244340309d4fc240f9aef55ff3a7/arc_agi/sandbox.py)
uses a host Python subprocess with a timeout; this is not our required container
isolation. Reuse the representation/feedback idea, not its executor or stopping
criterion. No upstream code was executed. Verified source hashes:
solve_coding.py 3f9bafef5b714da39aabb9889865c0ceb20ddca03abce108eb9e65461a82207a;
sandbox.py 525e5e5b8381e81c116cc6b18f586675e3bfce542a288ac0ff6787f8c7effa94.

[BARC](https://github.com/xu3kev/BARC) distinguishes code-producing induction
from direct output prediction and provides execution-based evaluation. It uses
fine-tuned models and ensembles, so its evidence cannot establish that our
unchanged Luna-medium route will improve merely by switching representation.

[CodeARC v2](https://arxiv.org/abs/2503.23145) studies interactive synthesis of
hidden functions using queries and differential testing. This is a possible later
environment, not an immediately qualified replacement: a differential testing
oracle may supply information unavailable to a real BED policy. Any adaptation
must charge and expose exactly the same observations to all policies. Nothing
in these sources establishes Bayesian calibration or monotonic depth for us.

## Concrete next architecture

Test ordinary Python transform(grid) functions against the existing nested DSL
on a fresh exact-schedule cohort, with a shared natural-language plan, equal
compilation/repair calls and fixed candidate slots. Preserve program diversity
after visible-example agreement; never return the first fit as a posterior.
Use the same observed examples, output schema, prediction-failure mass, frozen
targets and proper predictive scores. Keep Luna medium fixed to isolate the
representation intervention. Do not combine this with more observations,
larger reasoning or extra repair rounds in the same comparison.

Before paid calls: implement/test native-function execution inside our existing
no-network/no-key/read-only nonroot container, with fresh process per input,
bounded memory/CPU/output and permitted libraries explicitly fixed. Generated
code must never run in the host interpreter, source-generation mount or a process
holding labels from other examples. Test loops, filesystem/network attempts,
malformed/non-grid outputs, cross-input state and nondeterminism. Python AST
checks are interface validation, not a security boundary.

Only then freeze a new shared-plan representation protocol and fresh cohort.
Require unseen predictive coverage and meaningful answer disagreement, not just
better demo fit. The later scientific requirements remain identical online/branch
belief updates, coherent persistent worlds, a same-objective non-myopic gap,
compute-matched myopic and random controls, and paired sealed endpoint results.
No source here rescues an old cohort or authorizes depth testing.

This changes the next action from nested-DSL prompt repair to a controlled
native-code representation test. No new paid permission or tasks selected here.
Latest account balance23.238785711; delayed charges now posted, conservative day
remaining3.65706419 unchanged. No cluster/automation changes; goal unachieved.
