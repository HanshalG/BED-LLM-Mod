# Discovery rollouts must not grade their own confidence

## Decision and evidence

The preceding response restated an existing Luna result and made no scientific
progress. This turn independently replayed all saved transition endpoint scores,
then measured a previously unreported scoring reversal. It makes no model calls,
opens no new outcomes, and does not modify the banked experiment.

In case 3 (the only regeneration-versus-repeat recovery), the saved forecasts give:

| Updater | Forecast self-risk | Realized target half-Brier |
|---|---:|---:|
| Before observation | 0.244213 | 0.326968 |
| Structural insertion repair | 0.000000 | 0.656250 |
| Luna medium regeneration | 0.190651 | 0.113676 |
| Filtering / repeat-old-history | undefined (abstention) | 1.000000 |

Self-risk is half Gini impurity, `(1-sum(q^2))/2`, not Shannon entropy.
Both impurity and Shannon entropy are zero for the insertion forecast's point
masses. A self-risk-minimizing terminal scorer ranks insertion above regeneration;
the actual predictive loss ranks them oppositely. This is an updater comparison
on one opened case, NOT evidence of candidate-query ranking fidelity, a measured
bug in a deployed discovery planner, or an estimate of population calibration.
All four cases and five arms are retained in the accompanying JSON.

## Correct utility for an approximate belief updater

Let p be the simulator's target-output distribution conditioned on the complete
simulated branch, and q the forecast produced by the LLM-based updater there.
For half-Brier loss:

    E_(Z~p)[loss(q,Z)] = (1-||p||^2)/2 + ||p-q||^2/2.

Using `(1-||q||^2)/2` instead is correct when q=p, not for arbitrary regenerated
beliefs. Reducing the updater's uncertainty can increase its prediction error.
The implementation `expected_brier(reference, forecast)` preserves categories
present in either distribution, exposes the reference Bayes risk and excess
risk, rejects invalid mass, and assigns abstention the existing fixed penalty 1.
It neither smooths nor renormalizes bad forecasts. This is a standalone utility;
no frozen scoring code or previous result was changed.

For future discovery lookahead, integrate this proper loss over the simulator's
JOINT observation/target distribution and updater randomness. Condition p on each
simulated observation before scoring q. Equivalently, draw a simulator world,
simulate its answers, run the updater using only the public history, and score
the final forecast against that same world's reserved targets. Neither that world
nor its target labels may enter the updater prompt. Do not resample a world from
the regenerated belief to grade its forecast. Averaging outcomes independently
of the simulated observation also breaks the joint law.

This fixes a utility definition, NOT the simulator's missing support. In the
banked recovery case the revealing answer [7] still has zero probability under
the old pool. A proper scorer cannot recover a branch never simulated. An
arbitrary epsilon branch without a coherent conditional target law is likewise
insufficient. Exact full-posterior Bayesian prediction is a special case; any
benefit from generation here concerns approximate inference and representation,
not information created by computation alone.

## Source cross-check and architecture consequence

Murphy's MDA v2 expands its hypothesis space after predictive inadequacy, then
selects experiments by current-model information gain. Its ChemBench discussion
explicitly calls both acquisitions myopic. It supports executable LLM proposals
plus numerical inference, not an already-validated anticipatory discovery model.
[MDA v2, Sections 3 and 4.2](https://arxiv.org/html/2608.09696v2).

Socrates uses multiple-choice behavioral specifications represented by Hoare
triples. This provides a substantive motivation for behavior-level questions,
but the conference abstract alone establishes neither an open-support answer
model nor non-myopic gains. It does not validate our arbitrary eight Boolean
properties, whose original opportunity gate remains a null.
[PLDI 2026 abstract](https://pldi26.sigplan.org/details/pldi-2026-papers/36/Choose-Don-t-Label-Multiple-Choice-Query-Synthesis-for-Program-Disambiguation).

The next dependency is a prospective pre-answer JOINT simulator validation on
fresh program histories: answer coverage and held-out target proper loss, with
Luna generation conditioned on sampled answers, versus history-blind generation
and repeated-root controls. A marginal OTHER probability is not enough: it needs
executable witnesses or a validated conditional predictor. Freeze the interface,
all cases, semantic gates, budget, and endpoint sealing before calls. Do not
launch a depth grid, change the failed property's prior/budget to increase its
headroom, or claim monotonicity from this utility correction. Only after joint
transition fidelity should ordinary equal-budget h1/h2/h3 and compute-matched
myopic controls be evaluated.

## Verification and accounting

`tests/test_rollout_risk.py` plus existing predictive-score test: 12 passed in
0.16s. Includes exact manual decomposition with disjoint categories, calibrated
limit, confidently wrong rank reversal, branch conditioning, abstention and
invalid distributions. Scoped lint passed. The audit verified the terminal hash,
forecast seal, exact old score replay, and agreement of the new proper loss with
every saved realized score to 1e-12. JSON includes parent and outcome hashes.

Authenticated credits/usage/balance: 245 / 220.458278479 / 24.541721521.
London September 8 ledger validated: posted spend 0.081584485, conservative
recorded spend 0.117199285 including the old 0.04 uncertain reservation; remaining
4.882800715. This audit costs zero. No process remains running. Automation stays
paused; the full research goal remains active and incomplete.
