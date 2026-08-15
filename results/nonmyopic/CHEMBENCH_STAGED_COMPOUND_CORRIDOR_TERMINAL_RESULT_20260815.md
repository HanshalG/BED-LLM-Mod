# ChemBench staged-compound corridor terminal result

Date: 2026-08-15

## Disposition

`failed_closed`

The exact cohort, versions, action set, query seeds, transition, and thresholds in
`CHEMBENCH_STAGED_COMPOUND_CORRIDOR_PROTOCOL_20260815.md` are closed. No LLM
semantic gate, policy endpoint, or paper efficacy claim is authorized.

Model calls: 0  
Network calls: 0  
Cost: $0.00

## Result

Across all 27 held-out structure/difficulty cells:

| Planner | Mean terminal log-rate MSE |
|---|---:|
| d1 | 0.0734347172 |
| d2 | 0.0583444881 |
| d3 | 0.0565689035 |

- d2 versus d1: 20.55% mean reduction, 14 wins, 5 ties, 8 losses.
- d3 versus d2: 3.04% mean reduction, 9 wins, 13 ties, 5 losses.
- d3 versus d1: 22.97% mean reduction, 14 wins, 7 ties, 6 losses.

Per difficulty:

| Difficulty | d1 | d2 | d3 |
|---|---:|---:|---:|
| easy | 0.06324357 | 0.05137568 | 0.05023639 |
| medium | 0.09968993 | 0.08589249 | 0.08589249 |
| hard | 0.05737065 | 0.03776529 | 0.03357782 |

The gate fails because:

- d3 improves d2 by 3.04%, below the frozen 5% threshold;
- medium d3 is exactly equal to d2;
- d3 beats d1 on 14 cells, below 21;
- d3 does not improve all three difficulties.

All implementation-integrity conditions pass:

- every proposal is one complete three-particle atomic edit;
- no held-out truth particle is proposed;
- likelihoods, beliefs, and terminal losses are finite;
- immutable proposal replay is exact;
- planned risk equals held-out truth replay;
- d1/d2 and d2/d3 roots differ on at least one difficulty;
- d2 risk remains above numerical zero;
- dynamic d3 beats fixed-support d1 by more than 20%; and
- dynamic d3 is within the frozen full-support control bound.

Raw result:

- `results/nonmyopic/chembench_staged_compound_corridor/corridor-v1-20260815/RESULT.json`
- SHA-256: `cc6768fcd9895a8f40427236c57aa334f8dbb63ca3c6801f4a82bf22e18df3eb`

## Diagnosis

This is not a proposal-validity or planner-calibration failure. The dynamic method
has a strong d1-to-d2 gain and a favorable d3 paired direction, but the third level
does not add enough value under the frozen experiment economics.

Two properties explain the saturation:

1. Source noise is only 1%. Most categorical assay outcomes are nearly
   deterministic, so two adaptive decisions identify most of the useful
   structural information.
2. Every assay costs one unit and repetitions are forbidden. There is no planning
   tradeoff between a cheap screen, a precise repeated discriminator, and a later
   conditional assay. Medium therefore reaches an exact d2 policy fixed point.

The unrestricted evidence-ranked edit oracle also explores valid remove and
replace-core neighbors. That is scientifically legitimate and cannot be removed
after this result, but it means additional depth does not automatically correspond
to another truth-directed modifier addition.

## Prospective successor

Do not alter this gate. Freeze a new stochastic compositional ChemBench protocol
before evaluating it:

- use realistic log-rate noise;
- promote repeat count to a first-class action with proportional cost;
- compare cheap screening against repeated discriminators under one well budget;
- retain the public atomic compiler, held-out parameter particles, exact proposal
  replay, and paired target-query loss;
- use a cost-aware policy ladder where d1, d2, and d3 count future decisions while
  all policies receive the same total well budget;
- keep fixed-support, full-support, compute-matched myopic, and random controls.

This follows the MDA paper's stochastic NeuronBench result: when noise matters,
repeat allocation creates a genuine design lever rather than forcing every action
to have identical cost and precision.
