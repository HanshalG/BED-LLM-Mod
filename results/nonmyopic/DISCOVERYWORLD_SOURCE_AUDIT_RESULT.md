# DiscoveryWorld Source Audit Result

## Decision

DiscoveryWorld is a strong end-to-end scientific-agent benchmark, but the
released environment is **not ready for a paired non-myopic BED experiment**.
Do not spend OpenRouter budget on it yet.

The attraction is real: its action space is semantic and open-ended, actions
change a persistent simulated world, and task completion plus procedural
progress are scored by the environment rather than by the planning model.
However, the release does not define a belief prior or a BED objective, and
same-seed reset/replay is not observation-deterministic. A StrategyEIG wrapper
would therefore need both a new probabilistic task construction and a patched
counterfactual simulator before it could produce auditable evidence.

## Frozen Source

- Paper: `https://arxiv.org/abs/2406.06769`
- Code: `https://github.com/allenai/discoveryworld`
- Audited commit:
  `fd591323920be0d3786ef350955de1945aa571e5`
- License: Apache-2.0
- OpenRouter calls/cost: `0 / $0`
- OatML jobs: `0`

## What Qualifies

The release contains eight scientific task themes, three difficulty levels,
and five official parametric seeds, for 120 benchmark tasks. Normal and
Challenge tasks use a 1,000-step cap; Easy tasks use 100 steps. The API exposes
JSON actions, partial observations, deterministic task scorecards, and exact
task-completion flags.

Several themes contain genuine prerequisites or delayed experiments:

- Archaeology requires calibration on known artifacts before interpreting
  unknown radioisotope measurements.
- Lost in Translation requires gathering evidence about an alien vocabulary
  before acting on a compound instruction.
- Reactor Lab requires discovering an instrument-response law before setting
  unknown reactors.
- Plant Nutrients requires relating controlled soil measurements to an
  unknown growth rule.

Keeping the native action space would make an LLM useful for proposing
scientific hypotheses and executable experiments. Replacing it with a small
hand-written macro menu would remove that advantage and recreate the
ornamental-LLM problem.

## Blocking Mechanics

### No native BED state

The benchmark supplies one complete world per scenario seed. It does not
provide a prior over alternative scientific hypotheses, an observation
likelihood, or a reference myopic/non-myopic planner. The oracle
`criticalHypotheses` field is evaluation metadata and cannot be shown to the
policy.

Treating the five official seeds as particles is also unsafe: many instrument
readings are continuous, so one observation can fingerprint a seed rather than
represent uncertainty over the underlying scientific law.

### Reset/replay is not exact

A local headless audit instantiated Lost in Translation, Challenge, seed 0
twice, applied the same teleport action, and ticked once. Initial local UI
hashes and task-scorecard hashes agreed, but the post-action UI hashes differed.
The differences included mushroom descriptions attached to the same UUID
positions.

Explicitly resetting Python's global `random` state and NumPy's RNG did not fix
the mismatch. The source constructs an independent unseeded
`random.Random()` inside every `Object`; some objects use that generator during
world construction. The API has no snapshot/restore or branch-clone operation.
Its world-history export is a log, not a state loader.

This is acceptable for ordinary agent evaluation, but it breaks paired
counterfactual rollouts: two candidate strategies can inherit different hidden
world details even when reloaded with the same advertised seed.

### Endpoint and cost mismatch

Exact task completion and procedural progress are useful external endpoints.
The separate discovery-knowledge score, however, is produced by evaluating
free-form agent notes against hidden questions, so it is not an exact Bayesian
truth-recovery endpoint.

The official 100/1,000-step budgets make full end-to-end LLM evaluation
expensive. Before paying for such runs, the method would need to demonstrate a
strict lower-immediate/higher-terminal opportunity under an exact replayable
simulator.

## Required Successor

A scientifically distinct successor may proceed only after a zero-cost
implementation passes all of these gates:

1. fork or wrap the simulator so every object RNG is deterministically derived
   from the scenario seed and object UUID;
2. prove identical observations and scorecards for complete replayed action
   traces;
3. define a latent scientific-law prior that does not use finite seed identity
   as the hypothesis;
4. keep hypothesis and experiment proposal open-ended and LLM-generated;
5. use exact simulator observations and exact terminal task completion for
   evaluation;
6. demonstrate a strict oracle myopic-versus-lookahead root difference before
   any model call; and
7. compare paired non-myopic, myopic, naive-thinking, and random controls under
   the same action and model-call budgets.

Until those gates exist, DiscoveryWorld remains a promising future
LLM-scientist environment, not the next paid experiment.
