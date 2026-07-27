# Pi-Bench Release Source Audit Preregistration

Date: 2026-07-28

## Motivation

Pi-Bench is a strong candidate for the active LLM-native claim. Its public description
contains `100` tasks across five persistent twenty-session personas, `524` hidden
intents, an interactive user simulator, workspace/tool state, and graders for both
proactive intent recovery and final task completeness. The benchmark explicitly
rewards focused clarification of missing semantic requirements before execution.

Official source metadata observed before repository-file or task-value access:

- repository: `https://github.com/Simplified-Reasoning/Pi-Bench`;
- observed `main`/`HEAD`: `383910b1698758a198b86037c63a111c8edc32ad`;
- commit date: `2026-06-04`;
- license reported by GitHub: Apache-2.0.

## Scientific Admission Question

The exact release is admissible only if it exposes a target-blind counterfactual
clarification process:

1. the planner sees the natural request, permitted history, workspace, and tool state,
   but never the hidden-intent labels or graders;
2. the official user simulator can answer alternative free-form clarification
   questions from its private persona/task state;
3. the same hidden task and initial state can be forked across candidate questions;
4. an external released endpoint determines which hidden intents were inferred,
   elicited, and ultimately satisfied;
5. semantic question quality cannot be replaced by enumerating a complete released
   action/observation table;
6. at least some tasks contain path-dependent information acquisition: an early
   question, workspace read, or tool result changes which later clarification is
   useful;
7. paired myopic and non-myopic policies can share initial state and stochastic seeds;
8. the released system can run through an OpenAI-compatible API without using OatML.

The first experiment, if admitted, will isolate clarification quality before attempting
full artifact workflows. This makes the causal claim about sequential information
acquisition rather than general tool-agent competence.

## Access Boundary

The audit will clone only commit
`383910b1698758a198b86037c63a111c8edc32ad`, record its tracked-content digest, and
inspect licenses, schemas, configs, dependencies, runtime code, user simulator, state
forking, hidden-intent tracker, and graders.

Before opening task requests, hidden intents, persona memories, workspace values, or
grader criteria:

1. enumerate only persona IDs, session/task IDs, and release order;
2. verify exactly five personas with twenty ordered sessions each;
3. freeze each persona chronologically:
   - sessions `1-4`: mechanics;
   - sessions `5-10`: development;
   - sessions `11-16`: confirmation;
   - sessions `17-20`: retained;
4. store IDs, positions, counts, source commit, and hashes in a public manifest without
   task values.

Chronological splitting preserves the benchmark's persistent-workspace semantics and
prevents future-session information from entering earlier policies. If counts/order
do not match the paper or IDs cannot be separated from values, fail and amend before
content access.

## Zero-Call Structural Gate

After the manifest is frozen, only mechanics task values may be opened. No model call
is authorized by this document. Report:

- hidden-intent counts and dependency/reveal metadata;
- which private fields are visible to user simulator, agent, and grader;
- whether clarification questions are free-form or selected from a fixed menu;
- whether arbitrary candidate questions can be evaluated from identical forked state;
- whether user responses depend on full dialogue history;
- whether intent resolution is generated online or copied from scripted turns;
- whether Proc and Comp can be computed for modified trajectories;
- whether deterministic seeds or recorded simulator randomness support pairing;
- setup cost and whether a clarification-only runner can avoid unrelated tool actions.

The source passes only if:

- all eight admission conditions hold;
- at least `15/20` mechanics tasks contain at least three hidden intents;
- at least `8/20` require clarification rather than pure context retrieval;
- at least `5/20` contain an intent dependency, delayed reveal, or question-contingent
  follow-up that can create a depth-two choice;
- all mechanics tasks have an external intent-resolution endpoint;
- a mock or deterministic dry run can fork one task across two different questions
  without endpoint leakage.

Thresholds are conjunctive. A failure closes the exact Pi-Bench clarification route
before OpenRouter spend.

## Conditional First-Link Experiment

A full source pass authorizes a separate response-blind preregistration for the
development sessions:

- generate a target-blind semantic support of possible hidden requirements;
- propose a shared bank of clarification questions;
- compare a complete two-question non-myopic strategy against one-step expected
  information gain at matched generation and scoring compute;
- fork the official user simulator from common initial state for every candidate;
- measure realized hidden-intent resolution after question one and after question two;
- record support truth coverage, truth-anchored belief mass, selected-score fidelity,
  and top-one regret;
- include random-question and naive-thinking baselines.

Reasoning is reserved for the naive thinking baseline. BED policies use non-reasoning
models. No confirmation or full-workflow run is authorized until this first-link gate
shows positive paired endpoint gain and truth-anchored ranking.

There is no protected OpenRouter reserve. The full available balance may be allocated
by expected scientific value after validity gates. OpenRouter is the only model
backend; OatML, Slurm, and SSH are prohibited.
