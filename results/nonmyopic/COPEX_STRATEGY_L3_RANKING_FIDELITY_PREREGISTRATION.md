# COPEx L3 Posthoc Ranking-Fidelity Diagnostic

Registered on 2026-07-16 after the primary L3 endpoint was already finalized and
found to fail its preregistered intersection gate. This is an explicitly
exploratory, zero-cost mechanism diagnostic. It cannot change the L3 gate,
rescue the strategy-prior claim, or motivate a further paid run under the current
project authorization.

## Fixed Input

The sole input is the completed formal artifact
`results/nonmyopic/copex_strategy_l3_confirmation_recovery1/20260716/L3.json`
(seed 31003). The analysis reads no model credentials, makes no LLM requests,
does not rerun the policy, and does not select another trajectory, seed, endpoint,
or candidate pool.

## Question

At every recorded `strategy_eig` decision, did the analytic horizon-four score
rank the four proposed fixed plans in the same direction as their realized
information gain under the completed trial's truth and Gaussian noise stream?
This probes the first mechanistic link behind non-myopic selection: score fidelity,
not whether the selected root improved the primary receding-horizon policy metric.

## Replay Procedure

For each of the 30 formal trials and all 30 decisions:

1. Reconstruct the registered finite particle support (64 prior particles plus the
   true source), uniform initial posterior, and initial sensor position from the
   recorded seed schedule.
2. Replay the recorded selected action and observation to recover each decision's
   pre-decision posterior exactly; assert that replayed entropy agrees with the
   trace.
3. Recover the already-realized Gaussian standard-normal innovations from the
   selected history. For each recorded candidate plan, execute all of its recorded
   horizon steps from the same pre-decision posterior, using those same future
   innovations at the true source, and update the exact finite-support posterior.
4. Compare the recorded candidate EIG score with (a) that candidate's realized
   fixed-plan entropy drop and (b) its change in log posterior probability of the
   true source.

The counterfactual is a **fixed-plan** continuation. It is deliberately not called
deployed-policy regret: the live runner executes only the winning root and replans
at the next actual observation, whereas this diagnostic follows each candidate's
complete macro plan to test the original scorer's own horizon-level ranking.

## Frozen Outputs

The sole report will include, across the 900 decision cells and 3,600 candidate
evaluations:

- per-cell Spearman correlation between predicted score and realized fixed-plan
  entropy drop, aggregated as a mean of trial means with a trial-bootstrap 95% CI;
- the analogous score/truth-log-posterior correlation;
- pooled candidate Spearman correlations as descriptive context;
- predicted-top-one accuracy, predicted-top-one fixed-plan entropy regret, score
  margin, and within-cell score standard deviation, all aggregated by trial; and
- replay assertions, candidate counts, and root macro-kind counts.

The sign of the primary descriptive diagnostic is fixed in advance: positive means
the scorer's ranking tracks larger realized information gain; a confidence interval
that includes zero means the analysis supplies no positive score-fidelity evidence.
No numerical threshold, favorable subset, alternate horizon, or secondary score may
be substituted if this diagnostic is weak. Regardless of its result, the published
L3 conclusion remains that the preregistered non-myopic intersection gate failed.
