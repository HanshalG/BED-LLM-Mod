# Luna medium fixed-slot qualification: useful signal, gate not passed

Frozen implementation 36c380dd, protocol
e912a0709241b6918674ddb75c19fb384f4be42d931b91dd7d01c86757c33ddf.
One complete run, 24 accepted calls, $0.09445668, zero uncertain exposure.
All calls stopped normally, with 267..3459 reasoning tokens; no thinking-limit
exits, transport failures or adapter failures. Exact saved-response replay
reconstructs all requests, feedback, nested worker records, forecasts and results
with zero new calls. 35 focused tests passed before launch. No containers remain.

## Frozen result

Initial demo0 coverage was [true,true,true,false], passing the >=3/4 gate.
All five arms' forecasts were sealed before opening the 32 target outputs.

| Arm | Mean whole-grid Brier | Mean fixed-canvas Brier |
| --- | ---: | ---: |
| Aware refresh | 0.25 | 0.25 |
| Equal-call blind refresh | 0.50 | 0.50 |
| Initial support | 0.50 | 0.50 |
| Direct proposals only | 0.25 | 0.25 |
| Source-only symbolic prefix | 1.00 | 1.00 |

| Frozen task | Initial | Aware | Blind | Symbolic |
| --- | ---: | ---: | ---: | ---: |
| bdad9b1f | 0 | 0 | 0 | 1 |
| 2dee498d | 0 | 0 | 0 | 1 |
| 1caeab9d | 1 | 0 | 1 | 1 |
| 99b1bc43 | 1 | 1 | 1 | 1 |

The absolute gain .25, relative gain 50%, and nonworse-control conditions pass,
but only ONE task improves, below the frozen requirement of TWO. Therefore
qualification_passed=false and depth_authorized=false. This is not a full gate
pass, non-myopic result, or reason to relax the gate. The cohort is closed.
Four tasks, not 32 independent tasks, are the unit of task-level evidence.

## Mechanism and limitations

After all three observations, aware support had 1,8,2,0 consistent unique programs;
blind had 1,6,0,0; initial had 1,4,0,0. All consistent aware programs on the first
three tasks predicted all eight held-out outputs correctly. Empty pools produced
the declared unit failure forecast, so these Brier results are all-or-nothing
support/generalization outcomes, not evidence of smoothly calibrated uncertainty.

On 1caeab9d, initial proposals shifted objects toward the image's middle. Aware
repair proposals instead aligned objects using a color-ONE object's boundary.
Two programs then matched all demonstrations and targets. This is evidence of
useful observation-conditioned mechanism proposal in this one task, not evidence
that the model can predict its own future belief updates or choose useful queries.

Search ran 12 bounded workers and reused the banked source-only 128-prefix.
No new search expression failed conversion in this cohort. Direct proposals match
the aware score and surviving support counts: no measured benefit from search.
The shallow source-only prefix has zero consistent programs on all four tasks;
beating this weak finite-budget baseline does not show superiority to strong
classical synthesis. Two tasks were already solved, one benefited, one remained
unsupported. This mixture limits planning headroom and the evidence's breadth.

## Next decision

Do not start a depth sweep, reopen targets, rerun this cohort, or change thresholds.
Use saved artifacts for zero-call analysis of candidate disagreement, support
recovery and the unsolved task's failures. Any successor must prospectively justify
a broader opportunity for informative queries and a stronger usable comparator,
then independently qualify predictive support and branch fidelity. The observed
single-task support gain motivates that work but does not authorize it as a
confirmed positive. The full non-myopic research objective remains unachieved.

Closing authenticated credits/usage/balance:
$245 / $221.512733569 / $23.487266431. London Sep9 posted spend $1.05445509;
conservative recorded spend $1.09445509 including prior $0.04 uncertainty.
Remaining daily allowance $3.90554491; no filler or endpoint-informed retry.
Raw bank: `results/nonmyopic/rearc_slot_qualification_20260909/`.
