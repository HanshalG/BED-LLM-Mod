# Number Game Conditioned-Frequency Ablation

Date completed: 2026-07-30

## Decision

**Close the conditioned-particle route under the frozen conjunctive gate.**

This zero-call ablation kept the initial support extension-deduplicated and
uniform. It compared:

- propagated conditioned particles, where proposal multiplicity and retained
  parent particle mass carry through both observations;
- local refresh particles, where each refresh uses proposal multiplicity but
  retained parent hypotheses are reset to one copy each.

The frozen gate required both variants to improve canonical and independent
LLM-validation Brier at both observed-history stages. The local-reset variant
failed one held-out validation requirement, so no prospective policy run is
authorized.

## Propagated Diagnostic

The propagated variant was consistently positive:

| Cohort | One-observation reduction | Two-observation reduction |
|---|---:|---:|
| development first-link | 2.871% | 3.125% |
| development second-refresh | 3.198% | 3.261% |
| held-out dynamic-fixed | 3.487% | 3.228% |

All `96/96` trees improved at both canonical stages. In the held-out cohort:

- one-observation weighted-minus-uniform Brier was `-0.005386`, interval
  `[-0.006385, -0.004385]`;
- two-observation difference was `-0.004201`, interval
  `[-0.004924, -0.003498]`;
- independent Gemini validation Brier improved `1.594%` and `2.307%`, with
  `32/32` tree wins at both stages.

This is a post-hoc diagnostic inside a failed conjunctive gate. It does not
authorize a narrowed propagated-only prospective run.

## Local-Reset Failure

The local variant improved canonical Brier in every cohort:

- held-out one-observation reduction: `3.487%`, `32/32` wins;
- held-out two-observation reduction: `0.696%`, `21/32` wins, interval
  `[-0.001514, -0.000288]`.

But on the held-out independent Gemini bank after two observations, Brier
worsened by `0.113%`, with interval `[-0.000222, +0.000633]` and only `13/32`
wins. This fails the co-required cross-bank gate.

The result indicates that local proposal frequency is not sufficient. The
positive signal depends on sequentially propagating particle mass.

## Replay Boundary

The weighted first-stage best query matches the stored uniform-policy query on
only `36.5%`, `34.0%`, and `35.9%` of branches across the three cohorts.
Consequently, these are calibration ablations on stored paths, not full policy
replays.

## Artifacts

Result directory:

`results/nonmyopic/number_game_conditioned_frequency_ablation/number-game-conditioned-frequency-ablation-20260730T000826Z`

- `RESULT.json` SHA256:
  `6adc4c89b3caf6ef2c240fb482ad1a2c8381b53f8430b954f28a4df55b193f40`
- model calls: `0`
- cost: `$0`
- source trees: `96`

No source status or manuscript claim is reclassified.
