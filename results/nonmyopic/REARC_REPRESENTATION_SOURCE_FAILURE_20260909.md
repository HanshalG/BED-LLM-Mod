# Representation source qualification: closed before model execution

The six-task metadata cohort and exact schedule were frozen and pushed at
fc29ca61 before source execution. The one-shot source journal completed 46 of
66 channels, then failed on request index 46: task 8731374e, input mode, seed
46201. The process exited normally with the sanitized failure envelope:
`source_failed`, phase `verify`, error type `ValueError`, return code 1.
There was no stderr. Result SHA256:
f03f30422b6a8786a2dfc750dd7900a1f1b9013ab15c001a4f20be466a031a75.

The saved prefix independently replays with 46 completed channels and zero new
source calls. No complete public bank exists. No model call, predictive scoring,
repair, target-label opening, paid reservation, or depth experiment occurred.
No container remains. Do not rerun this schedule, replace a task or seed, or
salvage this cohort under the frozen protocol. All six selected IDs join the
closed set, now 50 IDs.

## What the failure establishes

Unlike the earlier unjournaled source failure, the exact request and failure
phase are known. It is not a timeout, API/provider problem or generated-program
failure. The worker places both reference execution/grid validation and the
generator-reference equality check in the verify phase. Its ValueError could
come from either; the saved record does not distinguish them. Do not label it
definitively as a mismatched output.

Read-only inspection of the pinned generator and verifier shows that the
generator embeds a structured patch inside random multicolor clutter, while
the verifier selects the largest connected object and repeatedly crops it.
This suggests a potential selection/cropping ambiguity but does not diagnose
the exact failed instance. No failed input or hidden output was regenerated.

## Next research decision

Repeated whole-cohort source failures are an avoidable experimental bottleneck,
not evidence against the representation hypothesis. Before another cohort,
prospectively design a source-qualified finite task pool: fixed metadata order,
fixed source request budget, exact per-task schedule and verification rules,
no model responses or performance-based selection. Report all source rejections
and explicitly restrict inference to the source-valid population. Freeze that
qualification mechanism before executing it; keep its retained outcomes sealed.
This is a new sampling protocol, not a rescue of the closed six-task cohort.
Do not simply weaken verification or trust unverified generated labels.

The existing shared-plan controller remains useful, but no paid representation
comparison is authorized from this bank. Its 24 focused controller/runtime/source
tests passed in 0.62 seconds; synthetic tests do not establish real source
compatibility. The new public-prompt preflight code was not executed after the
failed source gate, and does not establish prompt readiness.

Authenticated London time 2026-09-09T21:36:45+01:00: credits245,
usage221.761214289, balance23.238785711. Daily ledger unchanged, recorded spend
1.34293581 and unresolved reservation retained. This turn cost zero.
Previous turn: progress (controller implementation). Current turn: progress
(actual exact-schedule failure evidence and replay). Full goal remains unmet.
