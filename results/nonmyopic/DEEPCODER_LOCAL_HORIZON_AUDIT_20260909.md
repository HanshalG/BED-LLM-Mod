# Expanded Luna pools: no internal four-query horizon gap

Zero-call model-internal diagnostic on all four completed banked cases. Same
32 public target inputs, eight disjoint query inputs sampled with seeds
15100000+100*case+i for i0..7, four equally costly physical queries. Existing
ordinary-horizon planner replans after every outcome. Uniform restricted syntax
support, deterministic model outcomes, half-multiclass-Brier target risk.
Hidden source outputs remain unopened. This is not generalization evaluation.

| Case | Initial risk | h1 | h2 | h3 | Full B4 optimum | Random |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | .361107 | 0 | 0 | 0 | 0 | .006864 |
| 2 | .035807 | 0 | 0 | 0 | 0 | .002558 |
| 3 | .368800 | .002856 | .002856 | .002856 | .002856 | .021879 |

Receding open-loop h3 also matches these h1 values. Every panel completes;
initial root actions sometimes differ with horizon, but deployed terminal risk
does not. Mean h1/h2/h3/optimum risk .000714046; zero available improvement
over exact myopic at this budget on this restricted model. An exact myopic
control has no Monte Carlo estimator noise to remove by extra scoring calls.

Interpretation: local support expansion fixes a useful modeling bottleneck but
does not create a hard non-myopic decision problem. Case0 is already certain;
cases1/2 saturate; case3's residual cannot be reduced even by full-budget optimal
planning. This does not prove absence of horizon effects in the source grammar,
other costs, other budgets or open model discovery. It does rule out treating
this current panel as a promising depth-sweep experiment.

Do not shorten the budget, cherry-pick cases or silently change inputs to rescue
this null. A new task protocol should establish its own structural planning gap
prospectively, with a real motivating constraint (e.g. costly composite tests,
limited observations, or experiment-dependent access), classical myopic and
open-loop controls, and calibrated LLM-generated models. Alternatively, genuinely
anticipating model discovery requires a validated discovery transition, not
simply refreshing support after real observations.

Two focused tests pass in0.25s (zero-headroom equivalence and perfect query
resolution); scoped lint passes. All source supports replay exactly against
banked syntax lists. API calls/spend0; account unchanged, $.04 interrupted-call
uncertainty retained. Previous goal turn was progress; this diagnostic is new
negative structural evidence that prevents an uninformative paid depth grid.
Goal remains incomplete; no efficacy claim or automation restart.
