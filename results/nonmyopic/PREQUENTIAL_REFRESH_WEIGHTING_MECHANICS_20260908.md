# Prequential pipeline weighting mechanics

Implementation step following the exact banked refresh-overshoot diagnosis.
No new endpoint, held-out fitting, selected interpolation coefficient, paid call,
or gate authorization. The earlier experiments and their statuses are unchanged.

## Implemented update

`core/prequential.py` weights a fixed set of named, history-adaptive categorical
prediction pipelines. Example experts could be initial-support filtering and
retained refresh. It does NOT give each generated structure another prior vote.
Initial strictly positive normalized weights are supplied by the experiment; no
prior is inferred from the old target bank and no experiment default is chosen.

At each event, freeze every pipeline's normalized outcome distribution and the
current mixture BEFORE accepting the next observation. Upon outcome y, update:

    w_next[j] = w[j] * p_j(y | past) / sum_k w[k] * p_k(y | past)

Use exact rational arithmetic. Predictors may change structures after seeing y,
but those predictions can only affect the subsequent event's score. This is a
predictive mixture over adaptive algorithms, not posterior mechanism probabilities
or a selection-corrected Bayes calculation over generated structures.

The API requires the complete fixed expert set, immutable copied forecasts, one
outstanding forecast, one use per event, and the original forecast object when
scoring. Scoring a stale/reconstructed forecast or replacing forecasts before an
outcome is rejected. It records predictive probability, expert likelihoods and
posterior weights in an immutable receipt. No model count changes expert mass.

Zero likelihood eliminates an expert. If all weighted experts assign zero to the
observed outcome, the learner fails terminally rather than clipping, resetting or
accepting a different retry label. Any smoothing or expert-admission/revival rule
would need a separate prospective specification. The current implementation
supports 2-32 discrete outcomes and at most 32 named pipelines; it is not yet a
continuous ChemBench density adapter or durable experiment-ledger runner.

## Verification

43 focused tests pass in 0.57s across this mechanism and the shared rule/belief
modules. Direct new tests include independent likelihood-product weights, exact
normalization over all eight three-bit paths for history-adaptive predictors,
forecast copying/immutability, no repeated scoring, fixed expert identity and
terminal unsupported observations. A real executable-rule integration fixture
generates a broader hypothesis AFTER the first observation and verifies that it
can receive predictive credit only from the second observation, not the first.

These tests establish mechanics, not an efficacy or calibration result. Local
API ordering cannot prove that the caller never privately inspected an outcome.
A prospective runner must control data access and persist forecasts before the
outcome loader runs. The code also does not prevent a caller constructing a fresh
learner; the study runner must prohibit unauthorized resets.

## Scientific requirements still outstanding

Good sequential log prediction on selected queries need not imply good held-out
target prediction or better decisions. Only a fresh, prospectively frozen paired
proposal/update study can test transfer. Histories and query choices may differ
across deployed policies; never transplant another arm's future observations.

Before a paid block, specify task source, split, complete paired controls, explicit
pipeline priors, finite-positive likelihood treatment, proposal schedule, scoring
schedule, reset/admission rules and endpoint access. Measure target Brier/log loss,
mass assigned to initial/refreshed predictors, proposal count and inference cost.
Hold-out outcomes must not tune priors or mixture parameters. Preserve an original
uniform-refresh arm so any improvement can be attributed to weighting rather than
a simultaneous task or proposer change.

The full goal remains useful, non-myopic sequential BED with LLM-generated models
and ultimately anticipated discovery. This module addresses one diagnosed update
failure but is not that result, nor does it reopen any closed Number Game interface.
No cluster use or fresh account balance asserted; automation remains paused.
