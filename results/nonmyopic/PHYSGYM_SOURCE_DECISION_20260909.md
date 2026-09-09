# PhysGym: useful semantic-context design, not a ready BED experiment

Pinned source: https://github.com/principia-ai/PhysGym/tree/fe68079c0921029dde679ed44bc3192dd3b270ab

The README describes97 executable PHYBench-derived tasks and four levels of context
disclosure. We inspected the environment wrapper, research interface, evaluator and
execution helper, but did not download the task JSON, instantiate an environment,
execute benchmark code, or call a model. Counts and claimed task properties remain
README statements, not an independent task-level audit. License badge says MIT;
individual data provenance has not been separately verified here.

## Verified contracts

- PhyEnv loads a fixed task's Python function, executes it with supplied inputs,
  and exposes dummy variables/context metadata. The inspected wrapper does not
  define a Bayesian distribution over unknown structures or noisy observations.
  That is not proof that every task is deterministic or low-dimensional: their
  implementations remain unopened.
- ResearchInterface.test_hypothesis passes the true equation to the evaluator.
  Symbolic equivalence and an optional Gemini/OpenRouter judge return correctness
  feedback. Using this inside a BED rollout would introduce a privileged oracle
  channel; a clean adapter must exclude it. Other orchestration manages a separate
  test quota, so absence of a guard inside this method alone is not evidence that
  the entire benchmark permits unlimited oracle calls.
- The evaluator calls create_function_from_string with sandbox=True AND
  fast_local=True. The fast-local branch directly execs candidate source. Even the
  helper's sandbox branch execs the string in the parent to obtain its signature.
  Neither path should receive untrusted model code in this project. These are
  static source findings, not an exploit test or a complete security audit.

PHYSGYM_CONTRACT_AUDIT_20260909.json records hashes and exact call-site lines across
four source files. Static AST extraction executes nothing. One test(.08s) verifies
that a raising input is only parsed and that conflicting flags remain visible.

## Relation to previous evidence

PhysGym offers an interesting controlled semantic-context comparison, unlike the
random anonymous DeepCoder task. That could test whether learned knowledge improves
proposal coverage. It does not establish non-myopic opportunity or predictivity.
Its fixed function probes are not, by themselves, interventions that change which
experiments become available. We must not insert arbitrary unlock gates and claim
they came from this benchmark.

SciLaws already led this project to expensive continuous reference integration and
an unresolved fidelity/throughput problem. A fresh benchmark name does not solve
that. Rebuilding the same continuous hypothesis engine around PhysGym is not the
next default. The old SciLaws endpoint and thresholds remain unchanged.

## Architecture decision

Separate two questions before more environment implementation:

1. Can the LLM supply useful new executable mechanisms after a particular observed
   answer, beyond a fixed symbolic baseline and equal-width fresh sampling?
2. Does an independently calibrated simulator predict which experiment produces
   that improvement, and does the improvement leave useful terminal planning
   headroom rather than being exhausted by one greedy observation?

The lowest-cost next empirical design should test question1 with matched public
histories and semantic context, without launching a depth grid or whole benchmark.
PhysGym's context levels are a candidate source for this paired design only after
a strictly typed expression interface and a source-only development scope are
specified. Use the existing safe expression machinery if it covers the source;
do not execute arbitrary generated Python, expose true equations to the proposer,
or allow oracle-equivalence feedback. Independent target predictions, not exact
equation recovery or judge agreement, are the endpoint.

Before committing paid calls, choose development IDs by a fixed metadata rule and
inspect only those source implementations for an executable safe subset and valid
input distribution. Keep validation/confirmation source contents unopened. Record
source failures rather than replacing hard tasks. A small paid context-versus-blind
proposal gate could justify this route; a source/semantic failure closes it before
any planning implementation. Passing that gate alone would still not authorize a
headline or establish monotonically improving planning depth.

## Status

The previous goal turn produced the BoxingMoral audit; this turn produces an
independent PhysGym contract and a concrete boundary against repeating the SciLaws
engineering detour. Full goal remains active/unachieved. No paid calls; authenticated
account245/221.179955339/23.820044661 and daily remaining4.23832314 unchanged, old.04
uncertainty retained. Cluster, automation and historical runtime untouched.
