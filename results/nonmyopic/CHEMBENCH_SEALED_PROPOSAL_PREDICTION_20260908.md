# Sealed held-out proposal comparison: mechanics ready

The preceding goal turn made progress by connecting executable proposals to the
new planner. This turn implements the corresponding held-out scoring boundary,
not a new chemistry experiment. No failed endpoint or calibration-context audit
has been reopened. No model calls or source outcomes were requested.

## Comparison contract

`environments/chembench_mopen/proposal_prediction.py` requires exactly three
forecast arms for every case: `history_aware`, `history_blind`, and
`symbolic_search`. These are required input slots, not implementations or proof
that those proposer algorithms already exist.

All numerical fitters must have the same real-history hash, ordered fixed target
inputs and observation noise. The history-blind *proposer* withholds observations;
its numerical fitter still receives them. This isolates proposal usefulness from
the effect of withholding evidence during inference.

All forecast distributions are copied into one write-once JSON artifact before
any outcome-loader invocation. The evaluator verifies the exact expected hash
and validates the whole forecast panel before loading test outcomes. Missing or
failed arms, changed forecasts, duplicate cases/JSON keys, malformed weights or
target shapes, and incompatible fitting histories fail before that access.
Later in-memory model mutation cannot modify saved predictions.

Outcomes must cover every case and the exact same ordered target inputs. There
is no deletion of selected assays, difficult targets, or failed cases. An invalid
outcome set raises without returning a partial score report.

## Scores

- MSE against noiseless log1p target rates, with equal target weights within each
  case and equal case weights in the descriptive paired summary.
- Proper negative log predictive density on separately supplied noisy target
  observations, using the full posterior Gaussian mixture, not a Gaussian
  approximation to that mixture.
- Per-target probability-integral values and central 90% predictive coverage on
  those noisy observations. These are calibration diagnostics, not finite-panel
  certification of calibration.
- Per-case and per-target predictions and errors, plus paired aware-minus-control
  MSE differences. Negative differences favor the history-aware proposals.
- Numerical expression-evaluation work copied from each snapshot.

The output explicitly grants neither scientific-pass nor paid-call authority.
It does not insert an unfrozen significance or efficacy threshold. Proposal
tokens, fitting refinements and wall time are not inferred from expression work;
the eventual runner must record them to assess productive compute matching.

## Verification

67 focused tests pass in 1.71 seconds across sealed prediction, executable
snapshots, the original IR, expected-policy value and the closed context runner.
Scoped lint passes. The new tests independently check mixture density, MSE and
PIT; enforce outcome-access ordering and identical targets; reject missing arms,
tampering, schema defects and malformed outcomes; preserve saved forecasts after
model mutation; and exercise finite extreme-tail observations.

A constructed overconfident-wrong forecast has zero latent posterior variance
but worse held-out MSE and predictive density than the control. The scorer
therefore does not reward collapsing support merely for becoming certain.
This is a unit test of the measurement, not empirical evidence about an LLM.

## Limits and remaining plan

Hashes establish artifact identity, not that a researcher never inspected data
earlier. A prospectively frozen runner and separate outcome access are still
required. The adapter and scorer do not by themselves provide a model interface,
source semantic split, productive symbolic proposal search, parameter-integration
calibration, or a source-grounded planning opportunity.

In particular, the previous fixed-prior and calibration-context results remain
too weak to authorize a positive non-myopic claim or buy proposal calls on that
formulation. The next scientific decision must be an independently justified
source opportunity, not an assay/noise/seed sweep to rescue the closed panel.
After that, freeze the complete proposal/control runner, raw-response retention,
budget and semantic criteria before requesting any LLM responses. The eventual
paired policy study and powered fresh confirmation remain unexecuted.

Cost: $0. No cluster use, cleanup, new source endpoints or automation changes.
The full goal remains active and incomplete.
