# Luna-medium Python versus DSL: predictive improvement, support gate failed

The one-shot study ran from pushed commit caf1cfc3. All 30 Luna-medium calls
completed with normal stop, costing $0.15640007 with zero uncertain exposure.
Reasoning token counts ranged from261 to4142. All six Python pools contained
an observed-example-consistent program; forecasts were sealed before the60
authorized endpoint outputs opened. Exact terminal replay reproduced all30
requests, saved program evaluations, forecasts and metrics with zero new calls.
Result SHA256:6b17fa058fe6fd35decef71a8bf0979476c6f108a2f3f770afc7554574528ccd.

## Primary results

| Metric | Native Python | DSL |
|---|---:|---:|
| Whole-grid Brier, all60 outputs | 0.719411 | 0.934375 |
| Whole-grid Brier,12 query outputs | 0.717511 | 0.932292 |
| Whole-grid Brier,48 target outputs | 0.719886 | 0.934896 |
| Fixed-canvas Brier, all60 | 0.100731 | 0.424891 |
| Observed-example support | 6/6 | 4/6 |
| Actual query answers with positive probability | 1/12 | 0/12 |
| Actual target answers with positive probability | 4/48 | 2/48 |

Python improves aggregate whole-grid Brier by23.01% and fixed-canvas Brier by
76.29%. Five of six task-level whole-grid improvements exceed0.01. All six
Python tasks show predictive disagreement. The answer-support gate requires
8/12, however, and only1/12 passed. Thus qualification_passed=false and
depth_authorized=false. All frozen gates remain unchanged.

| Task | Python whole-grid Brier | DSL whole-grid Brier | Python query support |
|---|---:|---:|---:|
|855e0971|0.600000|1.000000|1/2|
|4258a5f9|0.810667|0.893750|0/2|
|bd4472b8|0.752066|1.000000|0/2|
|be94b721|0.713889|1.000000|0/2|
|bc1d5164|0.592188|1.000000|0/2|
|868de0fa|0.847656|0.712500|0/2|

The first task contributed two supported target outputs and the second another
two. The other four had no exact supported target output. DSL's two supported
targets were on4258a5f9. This is not broad exact-rule recovery.

## Critical interpretation

This is useful evidence for the representation-plus-interface intervention:
same plan, same model and observation, equal calls and attempted slots; Python
ran successfully on755 of756 cached code/input evaluations. It largely removed
execution failure, while exact prediction coverage remained poor. These cached
evaluations are dependent observations, not756 independent trials.

Lower whole-grid Brier alone does not imply correct branches are represented.
When truth has zero mass, the implemented normalized score is
(1+sum(p_j^2))/2. Spreading probability across several wrong grids can therefore
beat a concentrated wrong/failure prediction while still being unable to
simulate the realized answer. Fixed-canvas improvement supplies a complementary
cell-level signal, but includes padding and is not exact semantic correctness.
All60-output scores are diagnostic on six source-screened tasks, not a powered
claim about the full inventory. No monotonic-depth or non-myopic result follows.

## Next evidence

Keep the cohort closed to new calls or interventions. Use its saved programs,
plans and already opened endpoints for a zero-call error audit: distinguish
incorrect mechanisms, plan-to-code errors, near-correct grid mistakes and
spurious diversity. The next prospective change should address missing
predictive support, not increase planning depth or lower the8/12 gate. Test
whether additional public evidence or a different predictive observation
representation is scientifically justified before freezing a new experiment.
Any change in endpoint granularity must be defined before new responses and
cannot rescue this null. No unbudgeted follow-up is authorized by remaining funds.

## Accounting and verification

47 focused tests passed before dispatch. Offline replay is exact with new_calls0;
no candidate containers remain. Final authenticated credits245,
usage221.917614359, balance23.082385641. Conservative London-day spend1.49933588,
remaining3.50066412; the older .04 unresolved reservation remains retained.
The source rejection and restricted reference-valid population are reported in
REARC_SOURCE_QUALIFIED_REPRESENTATION_RESULT_20260909.md.
Previous turn: progress through source qualification. Current turn: progress
through a completed paid comparison. Full objective remains unachieved.
