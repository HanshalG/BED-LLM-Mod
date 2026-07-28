# SWE-Interact Release Source Audit Preregistration

Date: 2026-07-28

## Question

Does the official SWE-Interact release support an honest, irreducibly
LLM-native non-myopic BED test in which a first semantic plan or implementation
probe changes which hidden requirement the user reveals and therefore changes
the best second probe?

The candidate is motivated by a specific failure in Pi-Bench. Pi-Bench had
path-dependent LLM beliefs, but every untargeted question still revealed the
first unmet intent, making progress nearly action independent. SWE-Interact's
released user persona instead says to reveal private requirements in response to
specific plans, assumptions, and workspace changes, one issue at a time. Its
original repository verifier is an external endpoint.

No model call, task instruction, private user persona, reference solution, or
verifier content may be opened before the path-only split is materialized.

## Pinned Source

- repository: `https://github.com/scaleapi/SWE-Interact`;
- commit: `b32f98c3b8f76ca65e84341d1f30e5af7135f85d`;
- tree: `d2b16852aa541c780598283ceb0302a4dbe66aff`;
- release size observed before task-value access: 75 paired multi-turn and
  single-turn tasks, 25 each from DeepSWE, SWE Atlas Refactoring, and
  SWE-bench Pro;
- license: Apache-2.0;
- split seed: `24423`.

## Frozen Split

Within each of the three source families, sort task IDs by
`SHA256("24423:<family>:<task_id>")`, then by task ID. Assign:

| Partition | Per family | Total |
|---|---:|---:|
| Mechanics | 3 | 9 |
| Development | 7 | 21 |
| Confirmation | 8 | 24 |
| Retained | 7 | 21 |

The path-only manifest must be committed before mechanics values are opened.
Development may be opened only after the source gate passes. Confirmation and
retained values remain sealed until separately preregistered policy gates
authorize them.

## Release Integrity Gate

The path-only manifest must establish all of:

1. the checkout is clean and exactly at the pinned commit/tree;
2. exactly 75 unique multi-turn tasks exist, 25 per family;
3. every multi-turn task has a paired single-turn task;
4. every task package contains the public instruction, private user persona and
   server, repository environment, reference solution driver, final verifier,
   and all five released plan/implementation/handoff checkpoints;
5. the four partitions are a disjoint exhaustive cover with the frozen
   per-family counts.

Failure closes this release before task values or API use.

## Mechanics-Only Source Gate

After the manifest passes, open only the nine mechanics task packages and the
shared simulator/runtime source. All of the following are conjunctive:

1. the public initial instruction is materially incomplete relative to the
   simulator's private task goal on at least `7/9` tasks;
2. at least `7/9` tasks contain four or more atomic hidden requirements that
   can be checked against the external verifier or released rubrics;
3. at least `6/9` tasks contain a dependency or assumption structure where one
   requirement changes how a later requirement should be implemented or
   queried;
4. the released simulator conditions its reply on the agent's concrete plan,
   question, implementation, or workspace state, rather than replaying a fixed
   next sentence;
5. vague or irrelevant questions do not automatically reveal a hidden
   requirement or receive endpoint credit;
6. identical public histories can be forked before a model call, and private
   task text remains unavailable to the policy;
7. final correctness is determined by the original task verifier, not by user
   satisfaction or the policy's own generated hypotheses;
8. the semantic action/reply space is not reducible to a released finite
   candidate-response table available to a classical planner.

The audit will report exact counts and examples by task ID without publishing
private task text.

## Conditional First-Link Test

A source-gate pass authorizes a separately frozen mechanics-only serving test,
not development. The intended two-turn comparison is:

- GPT-5.4 nonreasoning generates a public belief over latent requirements and a
  shared bank of concrete plan/probe actions;
- myopic selection maximizes immediate expected requirement disclosure;
- depth-two selection values the branch-conditioned user reply, regenerated
  requirement support, and best second probe;
- both policies share every generated hypothesis, candidate, simulated reply,
  and token;
- a random shared-bank policy and a GPT-5.4 medium-reasoning naive policy are
  controls;
- the actual user simulator is forked from the same private initial state for
  each policy;
- primary first-link endpoint: external semantic recall of true hidden
  requirements after two user replies;
- required mechanism endpoints: changed roots, branch-sensitive regenerated
  support, truth-recall preservation, and predicted-versus-realized advantage
  correlation.

The test must first demonstrate, on mechanics only, that at least two candidate
roots elicit different valid requirement disclosures and that no generic prompt
receives automatic progress. Paid limits and development thresholds will be
frozen only after that zero-call/source gate.

BED policy calls use no reasoning. Reasoning is reserved for the naive baseline.
OpenRouter is the only paid provider. OatML, Slurm, SSH, and cluster resources
are out of scope. The full authenticated balance is available by expected
scientific value; there is no reserve.
