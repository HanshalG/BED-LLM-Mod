# RockSample[15,15] Scale Qualification

Registered 2026-07-22 before inspecting any BED value or trajectory on this
geometry.

## Purpose

Test whether the exact enabling-information mechanism, and conditionally the
LLM-Modulo StrategyEIG result, scale from 2,048 to 32,768 latent rock vectors.
RockSample[15,15] is a standard benchmark size, but current implementations
generate coordinates from a seed rather than sharing one canonical POMDPX file.

## Frozen Geometry

- Generator: `pobax/envs/jax/rocksample.py` at POBAX commit
  `a5e1d62d14e4efe783885b9d4f19cffa2a568eec`.
- POBAX `generate_map(15, 15, jax.random.PRNGKey(24098))`, JAX `0.11.0`.
- The generator's sampled flat indices are:
  `148, 34, 133, 92, 152, 25, 164, 84, 185, 118, 79, 138, 0, 44, 108`.
- Zero-based rock positions, in generated order:
  `(13,9)`, `(4,2)`, `(13,8)`, `(2,6)`, `(2,10)`, `(10,1)`, `(14,10)`,
  `(9,5)`, `(5,12)`, `(13,7)`, `(4,5)`, `(3,9)`, `(0,0)`, `(14,2)`,
  `(3,7)`.
- Start position: frozen left-center entry `(0,7)`.
- Diagnosis-only adaptation: retain the 32,768-state latent rock vector,
  deterministic motion, and distance-dependent checks; omit reward, sampling,
  and exit actions.
- Sensor half-efficiency distance remains `log(2)`, matching every earlier Rock
  Diagnosis result.

## Exact Qualification

- Fresh seed `24099`; 100 paired trajectories; 15 rounds; 5,000 paired bootstrap
  replicates; 2 trial workers.
- Exhaustive d1 and exhaustive receding-horizon d2 over every legal root and every
  positive-probability observation branch.
- Primary endpoint: paired entropy-AUC gain, d2 minus d1. The structural gate
  passes only if its 95% bootstrap lower bound is strictly positive and all
  pairing, legality, trace-length, and zero-LLM mechanics pass.
- Truth-log-posterior AUC corroborates only if its paired lower bound is positive.
  Final entropy, MAP accuracy, movement counts, scorer units, and initial exact
  values are secondary diagnostics.

No exact value, selected action, trajectory, or policy endpoint on this frozen map
was inspected before this protocol was written. A separate two-round runtime smoke
may verify memory and execution only and cannot change the geometry, seed, rounds,
endpoints, gates, or analysis. If the exact gate fails, no paid LLM run follows.
If it passes, the next loop must separately preregister a K4 branch-policy serving
smoke and paired StrategyEIG confirmation before spending.

## Frozen Command

```bash
python scripts/nonmyopic_rock_depth_oracle.py \
  --map 15-15 --num-trials 100 --num-rounds 15 --max-depth 2 \
  --seed 24099 --bootstrap-replicates 5000 --trial-concurrency 2 \
  --output-dir results/nonmyopic/rocksample_15_15_exact_qualification_20260722
```

## Registered Outcome

The exact gate passed all registered conditions. Exhaustive d2 improved paired
entropy AUC over exhaustive d1 by `+0.58884` nats (95% bootstrap CI
`[+0.58006,+0.59706]`; 100/0/0 wins/ties/losses). The truth-log-posterior AUC
gain was `+0.59301` (`[+0.54702,+0.63960]`), and the final-entropy gain was
`+1.38824` (`[+1.36546,+1.41002]`). All pairing, legality, trace-length, and
zero-LLM mechanics passed.

The initial exact d1 and d2 values were `0.00572` and `0.02985`. Greedy d1 made
no movement in 1,500 decisions, while d2 moved in 500/1,500 decisions, providing
the registered enabling-information mechanism rather than only a terminal metric
difference. An independent trace-level analyzer reconstructed every paired value,
bootstrap interval, action count, and gate from the raw report.
