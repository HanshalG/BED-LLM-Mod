# Gated Sensor v4 Informative-Terminal Result

## Decision

The preregistered 26B v4 smoke failed its primary matched-random gate. The terminal
menu correction repaired the sequential policy mechanics and brought StrategyEIG to
the exhaustive depth-two ceiling, but the LLM did not add measurable search bias over
the matched random proposal distribution. No Gated Sensor v4 formal run follows.

## Frozen Smoke

- Job: `106342`, `msc/oat14`.
- Model: `google/gemma-4-26B-A4B-it`, direct vLLM, non-thinking.
- Design: fresh seed `24118`, 4 paired trials, 8 rounds, K4, exact h2 scoring.
- Interface: v3 exact branch beliefs plus observation-producing terminal actions
  only.
- Requests: 55 physical, 52 accepted, 3 corrected retries.
- Usage: 260,608 prompt and 1,581 completion tokens; zero reasoning tokens, forced
  exits, API cost, or rollout-scoring LLM calls.

## Policy Results

Positive entropy-AUC gain means StrategyEIG has lower posterior entropy over rounds.

| Comparator | Mean gain | Paired 95% CI | W/T/L |
| --- | ---: | ---: | ---: |
| Shared-cell d1 | +1.1945 | [+1.0965, +1.2924] | 4/0/0 |
| Exhaustive d1 | +1.1945 | [+1.0965, +1.2924] | 4/0/0 |
| Matched random v4 | -0.0352 | [-0.1173, +0.0167] | 2/0/2 |
| Exhaustive d2 | +0.0039 | [0.0000, +0.0117] | 1/3/0 |

Truth-log AUC was also unresolved versus matched random: `-0.2318` with 95% CI
`[-0.6795, +0.0052]`. The four-pair smoke is not a powered equivalence test, but its
registered superiority condition clearly does not pass.

## Mechanism Audit

The zero-call audit reconstructed 28 reached proposal cells and 164 branch choices.

| Metric | v2 | v3 | v4 |
| --- | ---: | ---: | ---: |
| Overall immediate-EIG efficiency | .5364 | .6569 | .7113 |
| Measurement-root efficiency | .3200 | .5040 | .5887 |
| Measurement-root optimal rate | .1058 | .2788 | .2885 |
| Measurement terminal activation rate | .4904 | .2404 | .0000 |
| Measurement zero-EIG rate | .4904 | .2404 | .0000 |

V2 repeatedly activated A then B. V3 reduced but did not eliminate early panel
switching. V4 used exactly two activations and six precise tests in every trajectory,
matching the exhaustive d2 action-type budget. The corrected grammar therefore fixed
the simulator-policy mismatch it targeted.

The remaining result is more informative than another prompt iteration: once all K4
policies contain horizon-valid informative continuations, exact scoring makes the
matched random candidate set nearly exhaustive-quality too. The LLM's semantic
ranking is no longer the bottleneck and does not beat that control. This synthetic
task supports the exact non-myopic mechanism, but not an additional LLM proposal
advantage under the registered interface.

## Artifacts

- Preregistration: `GATED_SENSOR_V4_INFORMATIVE_TERMINAL_PREREGISTRATION.md`
- Raw run: `gated_sensor_v4_26b_direct_smoke_20260722/RESULT.json`
- Choice audit: `gated_sensor_v4_26b_direct_smoke_choice_mechanics_20260722/AUDIT.json`
- Choice summary: `gated_sensor_v4_26b_direct_smoke_choice_mechanics_20260722/AUDIT.md`
