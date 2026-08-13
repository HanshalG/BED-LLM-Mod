# RevengeBench callable-arena execution-opportunity V2 protocol

Date: 2026-08-13

Status: **frozen before selected source content or opportunity trajectories**

## Predecessor

V1 is infrastructure-inconclusive before outcomes because RoboCode's released
Java target policies cannot be counterfactually queried on frozen states by the
release's Python-surrogate offline evaluator. V2 changes only arena eligibility.

## Frozen cohort and mechanics

Use exactly the V1 manifest entries for BattleSnake, Halite, and HuskyBench:

- the same three hidden strategy hypotheses per arena;
- the same three selected executable probe policies per arena;
- the same three paired probe seeds and three disjoint endpoint seeds;
- the same native public state/action parser and `actions_distance` function;
- the same first-64-decision temporal cap;
- inverse temperatures `{0.5, 1.0, 2.0}` and a uniform three-hypothesis prior.

For every hidden target/probe/seed run, require two fresh deterministic arms.
Canonical target states, target actions, terminal result, normalized score,
trajectory length, and action-distance inputs must match exactly.

For each target trajectory, invoke all three candidate target policies on exactly
the retained target-visible states. Mean native action distance over decisions and
paired seeds forms `D_q[y, theta]`. Require finite distances in `[0,1]`, at least
three decisions per simulation, and diagonal self-distance at most `1e-9`.

Convert each distance matrix to the same proper likelihood:

```text
p(y=i | theta=j, q) = softmax_i(-beta * D_q[i,j])
```

## Frozen policies

Use the independently tested exact finite-support implementation:

- **depth two:** maximize expected terminal entropy reduction with an adaptive
  second unused probe for each first observation;
- **compute-matched receding myopic:** select the first probe by one-step EIG,
  then the second unused probe by one-step EIG after the realized observation;
- **fixed:** best nonadaptive ordered two-probe sequence;
- **random:** uniform average over ordered distinct two-probe sequences.

Tie-break by manifest order and never count a tie as a changed action.

## Endpoint separation

Use the unchanged three endpoint seeds and mechanics opponent. Endpoint states
must be disjoint from opportunity trajectories. For every changed arena, each
hidden target must have a strictly positive off-diagonal candidate distance and
the pooled off-diagonal mean must be in `(0.02, 0.98)`.

## Gates

V2 passes only if:

1. all nine hypotheses and nine probes compile and every paired run completes;
2. every fresh-arm pair is exact and every native parser/distance contract passes;
3. every arena likelihood is nondegenerate at every beta: root EIG is positive
   and below `log(3) - 0.02`;
4. depth two changes the first probe from receding myopic at every beta in at
   least **two of the three** arenas;
5. every changed arena/beta has depth-two gain at least `0.01` nats;
6. fixed and random controls are reported without a directional gate;
7. endpoint separation passes in every changed arena;
8. no selected source, provenance, raw trajectory, or prior outcome is
   serialized publicly; and
9. OpenRouter calls/cost and OATML cluster use are zero.

Fewer than two robust changed arenas is `no_horizon_opportunity` and closes this
route for the headline. A pre-trajectory runtime failure is infrastructure
inconclusive and cannot be reinterpreted as scientific evidence. This protocol
authorizes no model call or efficacy claim.
