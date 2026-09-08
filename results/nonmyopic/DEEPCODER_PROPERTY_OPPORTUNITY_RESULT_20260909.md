# One-bit experiment reference: monotonic but small effect

Numerical protocol frozen37a6ad5e before results. Four independent128-program
empirical-prior panels,8Boolean actions,4query budget,32full-output targets.
All exact finite-reference evaluations completed. No LLM inference or training.

| Planner | Mean terminal half-Brier risk |
|---|---:|
| Receding h1 | .350585843 |
| Receding h2 | .346116694 |
| Receding h3 | .343889774 |
| Receding openloop h3 | .351122128 |
| Full B4 optimum | .340726859 |
| Random | .385033386 |

h1>h2>h3 in risk on every panel. Aggregate relative improvements are1.27% and
0.64%, below the prespecified5% per step; numerical criterion fails. Full-budget
headroom is only2.81% over h1, so this reference cannot deliver two successive
5% gains. Do not lower the criterion after seeing the result, select panels,
or substitute an LLM comparison for the failed structural requirement.

The smaller directional effect remains descriptive evidence that one-bit tests
create some non-myopic opportunity. It is not an independently replicated
LLM-native benefit or full-grammar posterior result. The finite prior supplies
all hypotheses to an exact planner, so the LLM plays no role in this artifact.
This is a diagnostic of experiment semantics, not the desired final result.

The random menu provides one input for each heterogeneous property. Whether a
real task can justify a richer constrained test menu and useful larger horizon
gap requires a separately prospective design, not scanning seeds until a win.
Alternatively, anticipated model discovery remains promising but still requires
a validated predictive model for unseen outcomes. Neither is established here.

Boolean semantics test passed.08s. Initial lint found three one-line formatting
violations; corrected after measurement without changing semantics or rerunning
the scientific screen. Final lint passes. No API cost, account unchanged,
prior uncertainty retained. Previous goal turn progress; this turn measured a
new interface and banked its structural null. Goal remains incomplete.
