# RevengeBench Paired Execution Opportunity Protocol

Date: 2026-08-13

Status: **frozen after the source audit returned
`execution_opportunity_required` and before opening selected probe-policy source
or executing any opportunity target**.

## Purpose

Test whether executable semantic strategy hypotheses and active opponent-policy
probes create a real two-step first-probe advantage over a compute-matched
receding-myopic controller. This is a zero-model-call opportunity audit. A pass
would authorize only a separately frozen LLM semantic calibration interface.

## Frozen Cohort

Use exactly the three source-admission `opportunity` targets in each replayable
arena: BattleSnake, Halite, HuskyBench, and RoboCode. Each arena is an independent
three-hypothesis strategy class. Never pool incompatible game strategies.

For each arena, choose exactly three probe policies from its source-admission
`reserve` split by ascending:

```text
SHA256("revengebench-execution-probe-20260813:" + arena + ":" + target_id)
```

This selection was frozen before reading any selected probe source or trajectory.
The generated manifest binds only target IDs, salted hashes, public entrypoint
existence, and paired seeds.

Use three common random seeds per arena/probe/hidden-target combination:

```text
1 + uint32(SHA256(
  "revengebench-execution-seed-20260813:" + arena + ":" + simulation_index
)[0:8])
```

The same three seeds apply to every probe and hidden target in an arena.

## Observations And Likelihoods

For each hidden target `theta`, probe `q`, and seed:

1. run the target against the probe in a fresh deterministic runtime using the
   already admitted arena seed controls;
2. retain the target-visible state and target action at each decision;
3. cap each simulation at the first 64 valid target decisions by deterministic
   temporal subsampling if necessary;
4. offline-run every candidate strategy `theta'` on exactly those frozen target
   states using RevengeBench's public action parser and game-specific
   `actions_distance` implementation;
5. average action distance across decisions and the three paired seeds.

For each probe this yields a 3x3 matrix `D_q[observation_prototype, hypothesis]`.
Diagonal self-distance must be zero within parser tolerance. Convert it to a
proper discrete observation model for each frozen inverse temperature
`beta in {0.5, 1.0, 2.0}`:

```text
p(y=i | theta=j, q) = softmax_i(-beta * D_q[i,j])
```

The prior over the three strategy hypotheses is uniform. Entropy is measured in
nats. This soft likelihood avoids declaring every long exact trajectory a
perfect identifier while staying tied to the benchmark's native action distance.

## Policies

All policies may evaluate the same three first-probe likelihood matrices and all
remaining second-probe matrices. A used probe cannot be repeated.

- **Depth two:** choose the first probe maximizing exact expected terminal
  entropy reduction with the second probe selected adaptively for each first
  observation.
- **Compute-matched receding myopic:** choose the first probe by one-step EIG;
  after the realized first observation, choose the remaining probe with maximal
  one-step EIG. It receives the same likelihood matrices and candidate count.
- **Fixed:** choose the best nonadaptive two-probe sequence in expectation.
- **Random:** uniform over first probes and then uniform over remaining probes.

Tie-breaking is lexicographic manifest order. Report full utilities and margins;
never count a tie as a changed action.

## Held-Out Endpoint Separation

For each arena, run the same three hidden targets against the mechanics protocol's
fixed public opponent on three disjoint endpoint seeds derived with prefix
`revengebench-execution-endpoint-20260813:`. Endpoint states are never used in
probe scoring. Offline candidate action distances on those states must have:

- at least one strictly positive off-diagonal mean per hidden target;
- pooled off-diagonal mean in `(0.02, 0.98)`;
- no action-distance input shared with a probe trajectory.

This is opportunity/separation evidence only; it is not a policy endpoint result.

## Gates

The execution opportunity gate passes only if:

1. all 12 targets and 12 selected probes compile and complete all paired runs;
2. deterministic fresh-arm replay remains exact for every sampled run;
3. all action parsers return at least three and at most 64 valid decisions per
   simulation, with finite distances in `[0,1]` and zero self-distance;
4. all four arena likelihood families are nondegenerate at every beta: root EIG
   is positive and below `log(3) - 0.02`;
5. depth two chooses a different first probe from receding myopic at every beta
   in at least two of four arenas;
6. on every changed arena and beta, depth two has strictly greater expected
   terminal entropy reduction than receding myopic by at least `0.01` nats;
7. fixed and random controls are reported direction-agnostically;
8. held-out endpoint separation passes in every changed arena;
9. no target/probe source, provenance, raw trajectory, or prior released outcome
   is serialized in the public result;
10. OpenRouter calls and cost are zero.

## Decision Rule

- **Pass:** freeze a small LLM semantic strategy-hypothesis and likelihood
  calibration interface on fresh reserve targets before any response.
- **No horizon opportunity:** close RevengeBench for the non-myopic headline if
  fewer than two arenas have the robust changed first action and margin.
- **Infrastructure inconclusive:** bank exact missing runs if a pinned runtime
  fails before a complete trajectory. Never reinterpret it as scientific
  evidence and never weaken the gate.

No threshold, beta, probe, target, seed, parser, or endpoint may change after an
opportunity trajectory is observed. This protocol authorizes no model call and
no efficacy claim.
