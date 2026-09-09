# Successful fits still have no fixed-support planning opportunity

## New saved-record evidence

The closed scene qualification was independently replayed and its forecasts
reconstructed from cached program executions only. No new program execution,
source call, model call or actual future output was opened.

| Task index | Arm | Consistent unique programs | Distinct predictions on each of nine public inputs |
|---|---|---:|---|
|0|raw / inventory|0 / 0|unsupported, not certainty|
|1|raw|0|unsupported|
|1|inventory|11|1 on every input|
|2|raw / inventory|0 / 0|unsupported, not certainty|
|3|raw|4|1 on every input|
|3|inventory|10|1 on every input|

No successful pool has predictive disagreement. Within these frozen empirical
beliefs, expected half-Brier target risk is zero up to floating-point roundoff.
Any deterministic query generated from these pools has one possible answer,
so conditioning cannot change the posterior or improve that internal risk.
This is a zero-headroom fact about the model, NOT proof of actual accuracy or
absence of scientific uncertainty. Inconsistent pools are separately marked
unsupported, never assigned a misleading zero-risk success.

Another structural constraint: the qualification menu has only two candidate
queries. In a fixed-support deterministic model with an order-invariant Bayesian
update, a two-query budget exhausts that menu, so reordering both queries cannot
improve terminal risk. This does not rule out order effects of a path-dependent
LLM updater, nor was this qualification panel designed as the final depth study.
Neither issue is repaired by more Monte Carlo samples or a larger reasoning cap.

The audit JSON's source_result_sha256 is the shared canonical-JSON digest, not
the raw file-byte hash reported in the terminal report. Three unit tests cover
unsupported-versus-certain, syntax-versus-prediction diversity, and failure mass.

## Focused source comparison

[ReplaySCM v1](https://arxiv.org/html/2605.08197v1) evaluates executable Boolean
causal models under interventions. Table1 reports Hidden-order full-pool
HeldoutExact .292 for GPT-5.4, .536 for symbolic exact-search and .620 for
bnlearn+DSL. Its Alternative-SCM setting supplies a reference and asks for an
alternative plus separating intervention. These are useful contracts for
distinguishing formulas from predictive behavior, but not evidence of an LLM
advantage or non-myopic BED. Its generation procedure actively reduces ambiguity;
that can work against the uncertainty needed for our planning comparison.
Do not adopt its existing static evaluation as a shortcut to our headline goal.
This was a paper-only audit; no source instances or hidden answers were fetched.

## Architecture decision

Do not spend on another cosmetic grid-prompt change, simply widen the identical
program pool, or equate model entropy with remaining uncertainty. The current
fixed-support planning interface is not ready even on its successfully fitted
tasks. Better perception may help synthesis but alone cannot address this.

The next prospective route needs TWO independently demonstrated properties:

1. A usable joint predictive distribution over experiment answers and terminal
   targets, including meaningful alternatives after the initial history. Missing
   support must be visible; injecting independent uniform answer noise is not
   a coherent world law. Independent proposal sources may help, but must beat
   an equal-call alternative and pass held-out calibration before planning.
2. A real selection problem: more feasible experiments than the remaining budget,
   and answer-dependent complementary follow-ups under the SAME terminal loss.
   First measure its prevalence under a declared public/source prior without
   using actual evaluation outcomes. Keep saturated and unsupported cases in
   the denominator; do not select the winners of a post-hoc search.

For an M-open version, the simulator must invoke the same history-conditioned
proposer as deployment, while an independently qualified joint world model
generates possible answers and scores terminal predictions. Future regeneration
cannot be assumed to recover truth. A fixed-pool planner is a mandatory baseline,
not a substitute for that transition. Compare compute-matched myopic regeneration,
real-history-only regeneration, random queries and a strong classical search.

Before another paid route, inspect a substantially different source/measurement
contract against both requirements, not just whether Luna can fit it. An active
subset of passive synthesis examples needs its own opportunity evidence; a new
benchmark name or higher observed-fit rate is insufficient. No new paid route,
cohort rescue, threshold change, or efficacy claim is authorized by this audit.

## Accounting and state

Previous turn completed a paired paid experiment. This turn adds decisive
model-internal opportunity evidence and a focused primary-source comparison.
Account refreshed unchanged: credits245/usage222.308414519/balance22.691585481;
London-day conservative remaining3.10986396. Cost0. Automations remain paused,
no cluster use, all closed endpoints preserved. Full goal remains unmet.
