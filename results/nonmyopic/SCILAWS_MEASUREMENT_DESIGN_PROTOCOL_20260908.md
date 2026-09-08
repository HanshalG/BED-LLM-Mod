# Frozen development measurement geometry

This fixes a scientific design dependency, not a full experiment authorization.
The preceding source audit was concrete progress. This artifact is determined
only by the saved public metadata and restricted development input contracts.
No new simulator measurements or model responses have been generated.

Result SHA256:
`5e9f2bd902fa9de251cbe033bdd7dd4d5ad8fc79d870e80add7920ed92a18197`.
The builder binds metadata SHA `46a3ffcf` and state-contract SHA `f6f5a182` in
full. The saved JSON specifies all coordinates, weights and row counts.

## Geometry and budgets

For each used input take the intersection of its public training interval and
runtime admissible interval. Empty intersections fail; no replacement task.
Order dimensions by the saved runtime input order. Use log coordinates exactly
when the lower bound is positive and the upper/lower ratio is at least 100;
otherwise use linear coordinates. This is a prespecified geometry convention,
not a fit to any responses. Input transformation and response noise space are
different concepts.

Map Halton indices 1 through 8, bases 2/3/5, through those coordinates to define
eight query choices. Map indices 129 through 192 to define 64 prediction targets
with equal weights. There is no scrambling, seed search or target deletion after
queries. The same menus and targets apply to every arm and depth within a task.
This fixes a discrete approximation to uniform prediction in transformed input
coordinates, not the benchmark's original observational distribution. No claim
of optimal continuous design follows from this menu.

Initial evidence consists of the transformed centre and two axial points per
dimension at coordinates .25 and .75, other dimensions .5. Collect two independent
replicates per point, share the identical initial data across arms, then allow
four adaptive rounds with one new observation each. Repeated selection is allowed
and costs another row. The eight query choices are available on every round;
repetition must get evaluator-owned fresh noise, not a policy-supplied seed.

| Task | Log input axes | Initial rows | Adaptive rows |
|---|---|---:|---:|
| Baseball | None | 14 | 4 |
| Bird flight | None | 10 | 4 |
| Lake thermocline | Surface area | 6 | 4 |
| Battery ageing | None | 6 | 4 |
| Mars craters | Diameter | 6 | 4 |
| Spirometry | None | 10 | 4 |
| Volcanic column | Mass eruption rate | 6 | 4 |
| Wind turbine | None | 6 | 4 |

Budgets match within each paired task, not across dimensionalities. The common
initial observations must be counted in total information access and measurement
cost even if acquired only once and reused by the evaluator. Future h3 scoring
must optimize contingent actions and replan, not merely choose a root followed
by a greedy rollout. All arms get the same four adaptive observations.

## Consequences and limitations

The intersection rule changes the effective public support materially for some
tasks: bird mass is .009-.49 kg rather than extending to the runtime's 10.597 kg;
battery cycle index is 32-128 rather than 1-168. These differences were observed
before any responses. Do not expand the ranges later to search for a depth gain.

This is a synthetic point-query benchmark. Continuous inputs can include fractional
games, ages or cycle counts and combinations outside empirical joint support.
The runtime accepts continuous box inputs; this does not establish feasible real
interventions. Neither titles nor future paper text should imply a real laboratory
or causal intervention result. Any integer-only or joint-support experiment is
a separately designed follow-up, not an unlogged correction to these coordinates.

Six source tasks use linear residuals and two use log-space residuals. The true
conditional observation mean need not equal the hidden formula because empirical
residuals need not have zero mean. Before measurements, the remaining protocol
must distinguish clean-law recovery from prediction of future observations.
Hidden residual atoms must not become the agent likelihood. Initial replicates
give some noise evidence, but do not establish heteroscedastic calibration.

Still required before execution: source usage/attribution review; executable
structure and parameter priors; inferred observation model; precise terminal
estimand and task-normalized loss; hidden endpoint/noise streams; complete
h1/h2/h3, open-loop, productive-compute myopic and random policies; numerical
refinement, runtime caps and opportunity thresholds. A small fixed dictionary
may qualify numerical opportunity but cannot establish that LLM proposals are
necessary. Paid proposal gates and the stronger anticipated-discovery claim remain
separate unfinished dependencies. No execution permission is implied by this file.

## Verification

Six tests pass in .10s: dimensions 1/2/3, row counts, target weights, interval
intersection, log/linear behavior, known Halton values, irrelevant hidden-field
invariance, inadequate budgets and bound eight-task reconstruction. Independent
saved-artifact replay matches exactly; lint passes. Builder requires Python 3.10+
(tested 3.12); the system's older python3 failed on strict zip before writing an
artifact, then the pinned 3.12 invocation completed. No endpoint was opened.

Live authenticated account credits/usage/balance remain
245/220.376693994/24.623306006. The current Europe/London ledger validates $0
spent and a $5 ceiling; scientific paid authorization remains false. Automation
is paused. The full goal is not complete.
