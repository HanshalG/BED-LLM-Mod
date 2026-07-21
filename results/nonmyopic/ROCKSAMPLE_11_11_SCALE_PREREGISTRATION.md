# RockSample[11,11] Scale Qualification

Registered 2026-07-21 before any trajectory endpoint on this geometry.

## Purpose

Test whether the exact non-myopic mechanism and, conditionally, the LLM-Modulo
StrategyEIG result scale from eight to eleven latent rocks on the standard SARSOP
RockSample[11,11] benchmark instance.

## Frozen Geometry

- Source: `examples/POMDPX/RockSample_11_11.pomdpx` at SARSOP commit
  `d9141104392fd0a7b35327fdf7d40ef4b71a13ca`.
- Grid: 11 by 11; start position `(0,5)`.
- Zero-based rock positions, in benchmark order:
  `(0,3)`, `(0,7)`, `(1,8)`, `(2,4)`, `(3,3)`, `(3,8)`, `(4,3)`,
  `(5,8)`, `(6,1)`, `(9,3)`, `(9,9)`.
- Diagnosis-only adaptation: retain the 2,048-state latent rock vector, motion,
  and distance-dependent checks; omit reward, sampling, and exit actions.
- Sensor half-efficiency distance remains `log(2)`, matching the prior paper maps.

## Exact Qualification

- Fresh seed `24078`; 500 paired trajectories; 12 rounds; 10,000 paired bootstrap
  replicates; 16 trial workers.
- Exhaustive d1 and exhaustive receding-horizon d2 over every currently legal root
  (up to 15 actions) and every positive-probability observation branch. At the
  left-edge start, 14 are legal because `move-WEST` is excluded. The action order and
  tie-breaking remain frozen from the existing exact depth oracle.
- Primary endpoint: paired entropy-AUC gain, d2 minus d1. The structural gate passes
  only if its 95% bootstrap lower bound is strictly positive and all pairing,
  legality, trace-length, and zero-LLM mechanics pass.
- Truth-log-posterior AUC corroborates only if its paired lower bound is positive.
  Final entropy, MAP accuracy, action traces, scorer units, and initial exact values
  are secondary diagnostics.

No initial-state or trajectory endpoint on RockSample[11,11] was inspected before
freezing this design. If the exact gate fails, no paid LLM run follows. If it passes,
the next loop must preregister and pass a fresh ten-cell root-slot serving smoke before
any formal StrategyEIG launch.

The legal-action count clarification above was made after the first map-unit test and
before running the oracle. It changes no seed, trajectory, endpoint, gate, or policy;
the failed assertion exposed only that a boundary move is correctly omitted.

## Frozen Command

```bash
python scripts/nonmyopic_rock_depth_oracle.py \
  --map 11-11 --num-trials 500 --num-rounds 12 --max-depth 2 \
  --seed 24078 --bootstrap-replicates 10000 --trial-concurrency 16 \
  --output-dir results/nonmyopic/rocksample_11_11_exact_qualification_20260721
```
