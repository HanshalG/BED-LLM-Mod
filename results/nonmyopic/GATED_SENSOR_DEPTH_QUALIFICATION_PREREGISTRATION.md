# Gated Sensor Depth Qualification Preregistration

Registered: 2026-07-22

## Purpose

Test whether the paper's non-myopic mechanism generalizes beyond the spatial Rock Diagnosis family. The synthetic gated-sensor task has a discrete 32-state fault code, always-available weak screens, and zero-information panel activations that unlock high-fidelity tests. This stage makes no LLM calls; it qualifies the task before any paid policy experiment.

## Frozen Environment

- Hidden target: five independent binary fault bits under a uniform prior.
- Weak screen accuracy: `0.65`.
- Precise test accuracy: `0.95`.
- Three overlapping panels, each exposing six bit/parity predicates after activation.
- Activating a panel takes one round, emits only a null observation, and has exactly zero immediate EIG.
- Horizon: 8 rounds.
- Policy arms: exhaustive exact depth one and exhaustive exact depth two.
- Planning utility: expected terminal information over the active planning horizon.
- Evaluation endpoints: mean posterior entropy over rounds (entropy AUC) and mean log posterior probability of the true state (truth-log AUC).

## Confirmation Run

- Paired trials: `500`.
- Seed: `24091`.
- Bootstrap replicates: `10000`.
- Common random numbers: the two arms share each trial's hidden state and action-indexed observation uniforms.
- All likelihoods, posterior updates, action values, and outcomes are computed locally and exactly.
- No LLM or OpenRouter calls are permitted in this stage.

## Gate

Proceed to an LLM proposal-policy experiment only if all mechanics checks pass and both paired 95% bootstrap lower bounds are strictly positive:

1. depth-two minus depth-one entropy-AUC gain;
2. depth-two minus depth-one truth-log-AUC gain.

The primary sign convention is positive when depth two is better: lower entropy AUC and higher truth-log AUC.

## Exploratory Disclosure

Before registration, a 100-pair run with the frozen environment parameters produced an entropy-AUC gain of `+1.2511` nats and a truth-log-AUC gain of `+1.3071`; all 100 paired entropy differences favored depth two. This run was used only to verify that the constructed environment contains the intended non-myopic mechanism. The confirmation seed, sample size, gate, and implementation above are frozen before the formal run.

## Reproduction

```bash
python scripts/nonmyopic_gated_sensor_oracle.py \
  --num-trials 500 \
  --num-rounds 8 \
  --seed 24091 \
  --bootstrap-replicates 10000 \
  --output-dir results/nonmyopic/gated_sensor_depth_qualification_20260722
```
