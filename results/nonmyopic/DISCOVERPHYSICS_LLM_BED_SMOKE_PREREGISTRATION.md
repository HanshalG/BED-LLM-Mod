# DiscoverPhysics LLM-Native BED Smoke

## Status

Frozen before any DiscoverPhysics model response or simulator endpoint. This is
a one-pair serving and mechanism gate, not a powered efficacy claim. A failed
gate closes this interface without prompt, parser, world, score, or threshold
repair. A full pass authorizes only a separately preregistered development
comparison.

## External Environment

- Official repository: `SampsonML/DiscoverPhysics`.
- Pinned commit:
  `33b7fa9df96de9c35744efd181ca7e5a8dd60ad5`.
- Public benchmark world: `extra_dimensions`, but the world name,
  implementation, explanation rubric, and true law are hidden from the policy.
- Official N-body simulator and official executable-law evaluator.
- Observation noise: position standard deviation `.01`, seed `24365`.
- Two real experiment rounds. Every action is one experiment with ten
  measurement times; batching multiple experiments into one round is disabled
  symmetrically.

The external endpoint includes the benchmark's two default held-out
trajectories plus five frozen crossover-stress trajectories at radii
`.4`, `.75`, `1.25`, `3`, and `6`. Their full JSON SHA-256 is recorded by the
harness and they are never shown to the policy. The stress endpoint is needed
because the public world's defining feature is a short-to-long-distance
crossover.

Before calls, the official simulator passed all 84 upstream tests locally.
The extra-dimension and oscillator executors both completed direct CPU
experiments.

## LLM-Native Belief Tree

Model: `openai/gpt-5.4` through OpenRouter with reasoning explicitly disabled.
No reasoning trace is requested; reasoning is reserved for a future naive
thinking baseline.

At each real history, one exact-JSON call must generate:

- five distinct semantic force-law hypotheses and probabilities;
- four distinct root experiments;
- two predicted qualitative observation branches per root, with branch
  probabilities;
- a posterior over the current hypotheses for each branch;
- one branch-conditioned continuation experiment; and
- a fresh five-hypothesis terminal belief after that simulated continuation.

All probability vectors must sum to one within `.02`, then are normalized
before scoring. Every root and continuation is range-validated. Duplicate
hypotheses, duplicate root experiments, duplicate outcome branches, malformed
probabilities, extra fields, markdown, or trailing text fail closed.

Exact code computes:

- myopic value: current entropy minus expected one-step posterior entropy;
- depth-two value: current entropy minus expected entropy of the
  branch-specific regenerated terminal beliefs.

This makes semantic hypothesis generation, likelihood prediction,
experimental design, and path-dependent belief regeneration load-bearing.
There is no fixed hand-enumerated force-law support.

## Paired Control

The first tree is generated once and shared by both methods.

- Myopic selects the root with maximum immediate EIG.
- Non-myopic selects the root with maximum depth-two EIG.
- Frozen candidate order breaks ties.

Each root is executed in a fresh simulator with the same noise seed. At the
second and final real experiment round, each divergent history gets one fresh
belief tree, and both methods select by immediate EIG because only one action
remains. Each policy then receives one final non-reasoning call to emit exactly
one explanation and one executable `discovered_law`.

The shared initial tree is the compute-matched control: root support,
hypotheses, branch likelihoods, and token spend are identical at the only
round where the objectives differ.

## Frozen Gates

Serving/mechanics all must pass:

- exactly five OpenRouter requests;
- zero reported reasoning tokens;
- total run cost at most `$0.50`;
- all three trees and both final laws parse exactly;
- myopic and depth-two initial roots differ;
- the depth-two root has strictly lower immediate EIG;
- the depth-two root has strictly higher terminal EIG;
- the realized first experiments differ;
- both laws have finite MSE on default and crossover-stress trajectories.

The one-pair scientific screen additionally requires:

- depth-two crossover-stress MSE is strictly lower than myopic; and
- depth-two deterministic concept score is not lower than myopic.

The five-point concept score is frozen before responses: explicit recognition
of an extra dimension, compactification, long-range inverse-distance force,
short-range inverse-square force, and a crossover/two-regime scale.

Failure triggers no same-world prompt repair, selected candidate rerun,
reasoning-model rescue, score rescaling, or additional seed. A pass is only an
authorization for a fresh powered plan, not evidence by itself.

## Budget

- Projected smoke cost: `$0.42`.
- Hard run cap: `$0.50`.
- Total new-spend operating cap through Monday remains `$15`.
- Protected OpenRouter reserve: `$25`.
- OatML is not used.
