# Luna-medium named-plan qualification: complete, gate failed

Frozen implementation 082df5c6; source/cohort frozen earlier at42b8b145.
Completed all36calls, exact model openai/gpt-5.6-luna with medium reasoning,
OpenAI-only. Accepted cost $0.14476104, uncertain exposure $0, no retry.
All responses normal-stop, reasoning354..3894tokens; totals358419prompt and
70865completion tokens. There were no transport or plan-schema stops. Invalid
program batches and runtime failures retain their prescribed failure treatment.
76 focused tests passed. Full saved-response/execution replay is exact with
zero new calls. Terminal result SHA256:
82250825f92d239318503db9ce6c99c545c2027aea82357b7989ae8db56fa4a7.

## Frozen outcomes

Contrasting demo0 coverage4/6 passed, permitting all60outputs to open only after
all forecasts were sealed. Both arms consumed18calls with paired request seeds.
Lower scores are better. Query/target columns are whole-grid Brier.

| Arm | All whole-grid | Query | Target | All fixed-canvas |
|---|---:|---:|---:|---:|
| Contrasting | .681388 | .699796 | .676786 | .512375 |
| Ordinary | .735417 | .718750 | .739583 | .367447 |

Aggregate whole-grid gain .054029, about7.35%, with2/6task wins above.01.
However, the complete conjunction failed:

| Gate | Observed | Required | Result |
|---|---|---|---|
| Positive probability on actual query answer | 3/12 | >=8/12 | Fail |
| Tasks with nonfailure predictive disagreement | 3/6 | >=3/6 | Pass |
| Paired whole-grid score | .054029gain,2wins | >=.01gain,>=2wins | Pass |
| Nonworse fixed-canvas score | .512375 vs .367447 | <=control | Fail |

This is a completed exploratory null, not planner failure or efficacy evidence.
No sequential transition-fidelity or depth study is authorized by this result.
The cohort remains closed; no threshold changes, replacement or paid retries.

## Saved-data diagnostic

Counts below use only the banked forecasts and already opened labels. No extra
source executions or model calls. Query support counts positive probability,
not top-one accuracy. Target support is over eight held-out outputs per task.

| Task | Consistent programs C/O | Query support C/O | Target support C/O | Whole-grid Brier C/O |
|---|---:|---:|---:|---:|
| 1fad071e | 5/7 | 0/0 | 3/0 | .5600/1.0000 |
| 75b8110e | 0/4 | 0/0 | 0/0 | 1.0000/.8125 |
| 25ff71a9 | 5/5 | 2/2 | 8/8 | .0120/.0000 |
| e40b9e2f | 3/0 | 0/0 | 0/0 | 1.0000/1.0000 |
| 7b7f7511 | 7/2 | 1/1 | 3/3 | .5163/.6000 |
| f15e1fac | 0/0 | 0/0 | 0/0 | 1.0000/1.0000 |

Both arms cover exactly the same3queryanswers. Contrasting adds target support
on task1 but does not repair designated-query coverage. On task4, three programs
fit the visible example yet all their query executions fail. Demo fit and
program multiplicity therefore do not establish usable predictive uncertainty.
Task2's empty contrasting support substantially worsens the fixed-canvas score.
Task3 is already solved by the ordinary control; contrast adds uncertainty there.

The requested plan-slot assignments were sent in compilation/repair prompts,
but semantic adherence of each implementation to its named explanation has not
been independently established. Family names are not treated as evidence of
diversity; the measured disagreement gate uses actual nonfailure predictions.
Equal calls are not equal realized tokens or computation. Six tasks do not support
a powered efficacy claim or a causal claim that additional reasoning helps.

## Next dependency and budget

The interface correction worked operationally; executable predictive coverage
remains the bottleneck. Diagnose saved public compile/repair failures and
out-of-example generalization before a new prospective intervention. More depth
cannot sample the nine query outcomes with zero current model probability.
Do not infer all possible queries lack opportunity from these two per task.

Authenticated closing credits/usage/balance245/221.662048609/23.337951391.
Conservative London Sep9 day spend1.24377013; remaining3.75622987. Unused allowance
does not authorize an improvised rescue. No cluster or automation changes.
Research goal remains unachieved.
