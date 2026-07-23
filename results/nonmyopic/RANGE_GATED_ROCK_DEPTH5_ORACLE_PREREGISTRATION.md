# Focused-Prior Range-Gated Rock H5 Preregistration

Frozen before the fresh 500-pair qualification seed is executed.

## Motivation And Excluded Development

The corner-start h4 task established a three-move enabling route. This new exact
positive control asks whether a four-move route creates a genuine receding-horizon
h5-over-h4 effect.

Two development seeds are excluded from all formal endpoints:

- Seed `24232`, secondary `p_good=.10`: open-loop h5 value exceeded h4, but both
  receding-horizon policies executed the same route. This setting was rejected.
- Seed `24234`, secondary `p_good=.005`, 20 pairs: d4 repeatedly checked remotely
  while d5 moved to rock 6. Descriptive entropy-AUC gain was `+.269001`, 20/0/0;
  truth-log-AUC gain was `+.171382`. This seed is screen-only.

No formal seed response, truth, observation, endpoint, or bootstrap has been
observed.

## Frozen Environment

- Standard RockSample[7,8] rock coordinates.
- Start position: `(6,6)`.
- Range-gated sensor: remote accuracy `.55`, on-site accuracy `.95`.
- Target: the full eight-rock binary vector.
- Independent prior:
  - rock 6 at `(3,5)`: `p_good=.5`;
  - every other rock: `p_good=.005`.
- Initial entropy: `0.9135006422` nats. The seven secondary rocks retain about
  `.22035` nats, so the task is not a deterministic single-bit support.
- Eight real rounds, receding-horizon exact policies.
- Depths compared: d5 versus d4.

At the frozen prior, exact d4 initially selects `check-6` with total value
`.0197373562`. Exact d5 selects `move-NORTH` with value `.4946319372`, then the
registered prefix is:

```text
move-NORTH, move-WEST, move-WEST, move-WEST, check-6
```

The round-five check occurs on site at rock 6.

## Formal Protocol

- Fresh qualification seed: `24235`.
- Trials: 500 paired truths sampled independently from the frozen prior.
- Observation uniforms: common deterministic random numbers whenever policies
  perform the same check at the same position and repeat index.
- Bootstrap: 10,000 paired replicates.
- Primary endpoint: d4 minus d5 posterior-entropy AUC, so positive favors d5.
- Corroborating endpoint: d5 minus d4 truth-log-posterior AUC.
- Secondary descriptive endpoints: final entropy, final truth log probability,
  and final MAP accuracy.
- Exact decision cache: each unique `(policy depth, position, remaining horizon,
  belief)` is solved once and reused across trials. It does not approximate,
  prune, or share values across nonidentical states.

## Frozen Gates

The qualification passes only if:

- the paired 95% entropy-AUC lower bound is positive;
- at least 450/500 paired entropy-AUC differences are positive;
- the paired 95% truth-log-AUC lower bound is positive;
- all d4 initial actions are remote `check-6`;
- all d5 initial actions are `move-NORTH`;
- every d5 trajectory follows the registered five-action prefix and checks
  rock 6 on site at round five;
- truths are paired, all actions are legal, every trace has eight rounds, and
  no LLM call occurs;
- a separately seeded audit replays every truth, action, observation, posterior,
  metric, initial value, and producer comparison, then independently recomputes
  both positive bootstrap lower bounds.

Independent audit bootstrap seed: `24236`.

## Claim Boundary

Passing establishes an exact h5-over-h4 structural opportunity on an engineered
but non-degenerate prior. It does not establish an LLM proposal or policy result.
A hierarchical h5 LLM gate may be designed only after this qualification and
audit pass, under fresh seeds and a separate preregistration.
