# Prior rejection does not sustain sequential support

The prospective diagnostic frozen at `eeadc08e` completed all64 contexts and
returned `rejection_budget_inadequate`. Each context allowed2048 independent
prior draws to find16 programs matching its complete observed history.

| Observations | Filled16 particles | Empty | Partial (1--15) |
|---|---:|---:|---:|
| 1 | 10/16 | 3/16 | 3/16 |
| 2 | 6/16 | 4/16 | 6/16 |
| 3 | 2/16 | 8/16 | 6/16 |
| 4 | 1/16 | 9/16 | 6/16 |

Total work:107117 program draws,111744 history evaluations,9.376 seconds summed
across contexts. All64 rows are banked. No case was retried with a larger cap or
supplied with its true program. The interpreter and original input hashes were
verified. These16 fresh truths differ from the preceding audit's512 truths;
this is not a paired comparison against that audit. Four history lengths share
each truth and are not independent trials.

## Selected-subset scores

Only completed contexts had predictions fixed and then scored on the32 targets:

| Observations | Completed subset size | Half-Brier | Internal risk |
|---|---:|---:|---:|
| 1 | 10 | .30490 | .29315 |
| 2 | 6 | .27747 | .19877 |
| 3 | 2 | .14124 | .05896 |
| 4 | 1 | .04272 | .11548 |

These changing subsets are NOT a learning curve. Their numerical improvement
does not establish improvement with observations. One completed four-observation
context cannot establish calibration. Empty and partial sets were not scored as
completed inference.

## Next action

Prior rejection has a coherent conditional sampling law in the ideal iid model,
but this bounded implementation is inadequate for most longer histories. Failure
does not mean the observations are impossible: all histories came from programs
in the grammar. Do not enlarge this audit's cap or select favorable histories.

The next useful baseline is history-guided executable program search under an
explicit work budget, with target outcomes withheld. Reuse an existing synthesis
implementation where possible. Measure inference completion and independent
prediction before another planning grid or LLM experiment.

Search results are not posterior samples merely because they fit observations.
Unequal visitation, multiplicity and prior mass need an explicit interpretation.
An approximate candidate-pool predictor can be evaluated honestly without an
exact full-grammar Bayes claim. Apply that same standard to later LLM proposals
and the productive symbolic control.

The original finite-bank opportunity null is unchanged. Neither computation
failure nor the earlier support failure establishes an LLM advantage, positive
non-myopic efficacy or completion of the full plan.

## Verification

14 focused rejection/support tests passed in .27 seconds; scoped lint passed.
Tests cover full-history matching, duplicate mass, empty/partial results, prior
sampling with no observations, budgets, and propagated evaluator/time failures.
Artifact `DEEPCODER_REJECTION_AUDIT_20260908.json`, SHA256
`a4d8b2e575762b22b1593c71a6ea95a3a5c1406fb2e2209c7b95a3de1ef1c9c9`.
Every context records work, completion and history/accepted-program hashes;
completed cases also record prediction hashes. The process exited normally.

Cost:$0; model calls:0. Authenticated credits/usage245/220.376693994 match the
London Sept8 ledger. No cluster, protected runtime or automation changes.
No paid experiment is authorized. Goal remains incomplete.
