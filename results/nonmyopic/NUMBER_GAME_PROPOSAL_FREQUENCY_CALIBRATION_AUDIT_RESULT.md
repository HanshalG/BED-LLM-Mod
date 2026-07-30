# Number Game Proposal-Frequency Calibration Audit

Date completed: 2026-07-30

## Decision

**Close the frozen full-trajectory proposal-frequency weighting route.**

The retrospective gate required proposal multiplicity to improve canonical
Brier at the initial belief and after one and two observations in both
development cohorts and the held-out cohort. Initial, no-history multiplicity
did not generalize, so the result is `retrospective_calibration_null`.

No prior prospective result is reclassified.

## Primary Result

The initial prior was effectively neutral and inconsistent:

| Cohort | Uniform Brier | Weighted Brier | Relative reduction |
|---|---:|---:|---:|
| development first-link | 0.139470 | 0.139749 | -0.200% |
| development second-refresh | 0.139721 | 0.139659 | +0.044% |
| held-out dynamic-fixed | 0.141055 | 0.141293 | -0.168% |

The held-out tree-bootstrap interval for weighted minus uniform initial Brier
was `[-0.000453, +0.000898]`.

## Conditioned Signal

After observations, frequency-weighted particles were consistently better:

| Cohort | One-observation reduction | Two-observation reduction |
|---|---:|---:|
| development first-link | 5.299% | 4.438% |
| development second-refresh | 5.916% | 4.463% |
| held-out dynamic-fixed | 5.881% | 4.460% |

Every one of the `96` trees improved at both conditioned stages. In the
held-out cohort:

- one-observation weighted-minus-uniform Brier was `-0.009083`, with
  tree-bootstrap interval `[-0.010570, -0.007672]`;
- two-observation weighted-minus-uniform Brier was `-0.005805`, with interval
  `[-0.006629, -0.004986]`;
- the independent saved Gemini validation bank agreed, with reductions of
  `3.230%` and `3.798%`.

This is a post-hoc mechanistic observation, not a passed policy gate. It
motivates a distinct intervention that keeps the initial prior uniform and
uses particle mass only after history-conditioned LLM refreshes.

## Replay Boundary

The raw artifacts preserve both independently seeded Qwen proposal draws, so
cross-draw multiplicity is exactly recoverable. The audit concatenates valid
particles from both draws and, after observations, every consistent retained
parent particle. The matched control is the exact saved extension-deduplicated
support.

A full weighted-policy replay is impossible from the saved trees. Weighting
changes the best simulated second query on most branches; compatibility with
the saved query is only `28.1%`, `30.3%`, and `32.0%` across the three
cohorts. New prospective generation is therefore required before making a
policy claim.

## Artifacts

Result directory:

`results/nonmyopic/number_game_proposal_frequency_calibration_audit/number-game-proposal-frequency-calibration-audit-20260730T000134Z`

- `RESULT.json` SHA256:
  `385e1c5b40725a6dc354e5621b3728e445cc3eb265d56f1d8c29e560a70a3020`
- model calls: `0`
- cost: `$0`
- source trees: `96`, split into two development cohorts and one held-out
  cohort

The private source-response hashes are bound in `RESULT.json`; private raw
responses are not published.
