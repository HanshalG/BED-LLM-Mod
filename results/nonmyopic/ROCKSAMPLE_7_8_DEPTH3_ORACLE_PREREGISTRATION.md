# RockSample[7,8] Exact Depth-Three Qualification

Registered 2026-07-21 before any trajectory endpoint was generated.

## Question

Does canonical RockSample[7,8] diagnosis contain incremental three-step value beyond
the already-confirmed two-step enabling-movement effect? This zero-LLM qualification
is the gate for building a paid explicit depth-three LLM policy grammar.

A performance-only initial-state probe was run before registration to size the exact
enumeration. It evaluated no sampled truth or trajectory endpoint: exhaustive values
were `0.00919`, `0.06928`, and `0.69315` nats at depths one, two, and three. This
confirms a candidate two-move-then-check mechanism and tractable runtime; it is not
confirmation evidence.

## Frozen Design

- Map: canonical diagnosis-only RockSample[7,8].
- Seed: `24075`.
- Trials: `500` paired truths.
- Rounds: `10`.
- Arms: exact exhaustive d1, d2, and d3 over every legal action and every nonzero
  observation branch.
- Pairing: each arm uses the same trial truth and deterministic common-random-number
  observation uniform keyed by trial, position, rock, and repeat count.
- Endpoints: mean post-round posterior entropy AUC is primary; mean true-state
  log-posterior AUC corroborates. Final entropy and MAP accuracy are secondary.
- Inference: `10,000` deterministic paired bootstrap replicates.
- No LLM, generated proposal, learned likelihood, Monte Carlo rollout, reward,
  sampling, or exit action is used.

## Gate

Positive values favor the deeper arm. Qualification passes only if the 95% paired
bootstrap lower bound for d3 minus d2 entropy-AUC gain is strictly above zero. The
d3-minus-d2 truth-log-AUC interval must also have a strictly positive lower bound to
count as corroboration. D2 minus d1 is reported as a consistency check but is not a
substitute for the primary gate.

A failed gate stops depth-three LLM engineering. A passed gate authorizes a separate
preregistered serving smoke and paid policy comparison; this oracle result cannot by
itself establish that an LLM proposal set captures the extra value.

## Frozen Command

```bash
python scripts/nonmyopic_rock_depth_oracle.py \
  --map 7-8 --num-trials 500 --num-rounds 10 --max-depth 3 \
  --seed 24075 --bootstrap-replicates 10000 --trial-concurrency 16 \
  --output-dir results/nonmyopic/rocksample_7_8_depth3_oracle_20260721
```
