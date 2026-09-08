# Public execution feedback: complete paired screen, no useful gain

Frozen executor2ea866d5, unchanged protocol145dd803. All24 Luna-medium calls
completed on eight fresh three-observation cases; no retry, fallback, schema
failure or transport error. Every endpoint forecast preceded target loading.

| Arm | Mean target half-Brier | Zero-mass targets /256 | Endpoint support |
|---|---:|---:|---:|
| Initial pool | 0.208059321 | 47 | 7/8 |
| Initial + feedback revision | 0.208094152 | 47 | 7/8 |
| Initial + context-only revision | 0.208146457 | 47 | 7/8 |

Feedback improves over its revision control by only0.000052305 mean Brier
(about0.025%), far below the frozen10% requirement. Both revisions are slightly
worse than the initial pool in aggregate. The exact feedback-control comparison
has six ties, one tiny win(case2,0.000697087), one tiny loss(case0,0.000278646).
No pair exceeds the required0.01 improvement. This is not a powered equivalence
claim; it is a failure to demonstrate useful improvement on the complete screen.

| Case | Initial Brier | Feedback Brier | Control Brier |
|---|---:|---:|---:|
| 0 | 0.000281 | 0.000559 | 0.000281 |
| 1 | 0.437500 | 0.437500 | 0.437500 |
| 2 | 0.000000 | 0.000000 | 0.000697 |
| 3 | 0.000000 | 0.000000 | 0.000000 |
| 4 | 0.186378 | 0.186378 | 0.186378 |
| 5 | 1.000000 | 1.000000 | 1.000000 |
| 6 | 0.040316 | 0.040316 | 0.040316 |
| 7 | 0.000000 | 0.000000 | 0.000000 |

Raw compatible/proposed counts: initial30/37, feedback39/52, control39/48.
These are different denominators, not a fit-rate gain for feedback. In case5,
initial0/1, feedback0/8, control0/8 fit the displayed examples; exact execution
feedback did not recover a compatible program. All arms abstain there, with
the fixed penalty1. Cases1/6 retain15 additional zero-probability targets, so
aggregate predictive NLL is infinite for all arms.

## Decision

Frozen gate fails coverage(7<8), relative predictive gain(<10%), and paired
wins(0<3); nonworse zero-mass count passes only by equality. No joint simulator
test or depth sweep is authorized by this result. Close this exact feedback
interface without retries or larger-budget rescue.

The previous public audit's observed contradictions were real, but the tested
feedback treatment did not fix the important failures. More generated programs
or more explicit correctness information did not materially change predictions.
Do not keep stacking proposal-prompt patches and call it progress toward deeper
BED. The next architectural decision needs to separate source-prior mismatch,
missing behavioral support and numerical reweighting, or establish a genuinely
LLM-native semantic context with predictive support. Those are hypotheses to
investigate, not a new environment or paid run authorized by this report.

This does not refute LLM-based non-myopic BED in general. It narrows what has
actually been tested: an untrained Luna proposer/reviser, uniform-syntax sampled
program worlds, restricted syntax-prior pool weighting, and this precise public
execution feedback format. The original non-myopic scientific objective remains
unchanged and incomplete.

## Verification and accounting

Independent replay verified24 request bodies/seeds, both revisions' identical
original candidate context, exact public-only feedback, source programs and
execution/fit/expansion statistics, prior-weighted forecasts, source target
identity, endpoint seal and scores/gates, and all receipts. Terminal SHA:
b85bd72c3dd978c91f1260690b99d25f541d0ff7f5aa7f5eae60e81d468c8f9e.
Forecast SHA86888a21a982e22af534a7770973a3e9541a59ec788d0b44ab8176d3f2ce1282.

Prelaunch12tests2.13s; replay access guards2tests0.33s; scoped lint passed.
Paid and replay processes exited normally. No historical experiment changed.

Actual cost0.06580485, no new uncertain exposure. Authenticated cumulative
credits/usage/balance245/220.663020549/24.336979451. LondonSept9 posted total
0.20474207; conservative recorded0.24474207 includes the old0.04 uncertainty,
remaining4.75525793. No cluster or automation launch. Goal active/incomplete.
