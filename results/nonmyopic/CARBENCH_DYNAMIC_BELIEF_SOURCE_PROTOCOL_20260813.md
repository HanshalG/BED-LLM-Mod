# CAR-bench dynamic-belief source protocol

Date frozen: 2026-08-13

Status: **value-blind source audit only; zero model calls**

## Scientific target

Use CAR-bench disambiguation tasks as latent natural-language intent worlds. The
LLM must maintain and regenerate semantic hypotheses about the user's intended
action and parameters while choosing between environment/preference tools and
direct user clarification. The desired first-link result is a depth-two query
that improves the quality of the next regenerated belief state and eventual
state-changing action relative to a compute-matched myopic query.

CAR-bench is relevant because its native policy requires internal resolution
from context when possible and user clarification only when multiple options
remain. It is not automatically a non-myopic BED environment: many tasks may
reduce to one direct clarification followed by execution. Source admission
therefore authorizes only a later mechanics opportunity gate.

## Immutable sources

### Code

- repository: `https://github.com/CAR-bench/car-bench`
- commit: `9ed387d8de2dac20e5227d8e949bd33d20041fc5`
- tree: `9f0d800b901f94277b2c9e80333a47bc5ac345eb`
- license: MIT

The audit binds hashes of the environment, task type, user simulator, reward,
tool registry, and wiki files used by the disambiguation contract.

### Data

- repository: `https://huggingface.co/datasets/johanneskirmayr/car-bench-dataset`
- commit: `1fcf24ad802c42e04a0d8fe05b5ca0d481a4e7af`
- config: `tasks_disambiguation`
- official train count: `31`
- official test count: `25`

The Hugging Face repository must be cloned at the exact commit. Moving cached
revisions are forbidden.

## Value-blind gates

The source passes only if all gates hold:

1. Code and data repositories exactly match the commits above.
2. The official `tasks_disambiguation` config exposes exactly train and test
   populations of 31 and 25 rows.
3. Every row in both splits has one exact shared schema matching the released
   `Task` fields required for disambiguation.
4. Task IDs are nonempty and unique across both splits; no train/test overlap.
5. Every row's task type is exactly `disambiguation_internal` or
   `disambiguation_user`, and both types occur in train and test.
6. Required structural fields are present and nonempty: persona, instruction,
   context initialization, action sequence, disambiguation note, and the one
   task-type-appropriate disambiguation element.
7. The released environment maps tasks into a mutable stateful tool environment,
   and the released user simulator distinguishes internal from user
   disambiguation with explicit failure behavior.
8. The released reward checks final state, intermediate mutations, required
   information tools, tool errors, policy errors, and conversation failure.
9. A deterministic value-blind split of the official 31-row train population is
   complete and disjoint: 6 mechanics, 10 opportunity, and 15 development.
   The official 25-row test population remains confirmation.

Train case identity is SHA-256 of canonical compact row JSON. Train rows are
ordered by `SHA256("carbench-dynamic-belief-v1|" + case_id)` before allocation.
Public artifacts contain only aggregate counts, schema field names, type counts,
and hashes of complete ordered ID lists. No individual ID, persona, instruction,
context value, action, disambiguation note/element, user answer, or endpoint is
serialized.

## Required mechanics gate

Only the six mechanics rows may be opened after this source boundary is pushed.
Before any paid call, a separate protocol must require:

- native completed-handshake replay with deterministic context and exact tool
  outputs;
- at least two valid information actions before the first state mutation;
- a dependent information structure in which an answer/tool result changes the
  valid or useful next information action;
- at least three semantic intent/parameter hypotheses with nontrivial truth
  coverage;
- strict answer-obedience and repeated-response calibration for any LLM user;
- answer-conditioned support regeneration and an explicit fixed-support control;
- different depth-two and compute-matched-myopic roots with positive oracle-linked
  first-link value on at least three of six mechanics tasks;
- paired common-random-number terminal state evaluation and random control;
- complete candidate-score banking before ground-truth actions or endpoints are
  opened.

Failure closes this exact CAR-bench construction. No task replacement, subset,
prompt repair, threshold relaxation, or test inspection is allowed.

## Accounting

- OpenRouter calls: `0`
- OpenRouter cost: `$0`
- OATML cluster use: none
