# Matched categorical prediction measurement

Implemented a no-network scoring path for history-aware, history-blind and
symbolic-search programs. No new source programs or model responses were drawn.
The pending experiment-order approval remains unanswered; no paid authorization
is inferred from the current goal continuation.

## Forecast construction

`environments/program_induction/prediction.py` checks every proposed candidate
against the SAME actual history, including the history-blind arm. It retains
unique canonical identities that fit every observation and records proposed
and retained counts and interpreter evaluation work. An empty compatible pool
raises explicitly; it is not replaced by a perfect or default forecast.

Retained candidates have uniform weights. This is an explicit computational
finite-pool convention, not a claim of full-grammar posterior probabilities.
The runner remains responsible for trustworthy canonical identities, bounded
pure interpreter callables and real-history provenance. Arbitrary model output
code is not an authorized callable.

Predictions use exact bounded output categories, including ERROR. Target inputs
are fixed across all arms and never deleted based on selected queries. The
module provides no access to true programs or target outputs while building
forecasts. Actual proposer/search implementations remain responsible for not
inspecting those targets earlier.

## Seal and score

Every case must contain all three arms with identical ordered target inputs and
real-history hashes. Counts, normalization, candidate identities, categories,
and evaluation work are validated before write-once forecast serialization.
The returned SHA256 binds saved bytes. Later mutation of live forecast objects
does not alter the saved predictions.

Scoring verifies the expected hash and complete forecast schema before invoking
the outcome loader. It validates complete case/target coverage before returning
any scores. Tampered files or incomplete forecasts do not open target outcomes.
There is no partial-panel fallback.

Outputs are per-case/arm half-multiclass Brier and negative log predictive density.
Unseen actual categories produce finite Brier but infinite log loss, represented
by `nll=null` with `nll_is_infinite=true` and a zero-mass count. No epsilon is
inserted. No significance threshold, scientific pass or paid permission is added.

## Verification

40 focused prediction/proposal/search/rejection tests pass in .70 seconds; scoped
lint passes. Constructed fixtures cover hand-computed mixture scores, a
confident-wrong forecast, explicit empty pools, duplicate identities, unequal
targets, missing controls, write-once behavior, tampering, and post-seal mutation.

A complete fixture compiles two DSL proposals, runs actual pinned CrossBeam,
builds all three forecasts on the same history and target inputs, seals them,
and then loads the fixture truth. It demonstrates integration, NOT an LLM
generating those programs or empirical superiority over search.

## Remaining work

A prospective comparison still needs a reviewed case protocol, raw-response
retention, all-case failure accounting, model/catalog verification, account-wide
request reservations and explicit dispatch authorization. Hashes establish
identity, not proof that data were never inspected. This helper supplies no
general posterior calibration or policy efficacy result.

The full numerical-reference-first plan remains in force unless the requested
proposal-only sequencing change is explicitly approved. No model calls/$0 spend;
authenticated credits/usage245/220.376693994 match the London Sept8 ledger.
No cluster, protected runtime or automation changes. Goal remains incomplete.
