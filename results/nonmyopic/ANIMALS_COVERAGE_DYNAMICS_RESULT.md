# 20 Questions Coverage-Dynamics Screen Result

**Status:** completed exploratory mechanism screen; not a policy or endpoint
result.

The first valid run is
`animals-coverage-dynamics-gemma26b-recovery-20260718`, using non-thinking
Gemma 4 26B A4B. It collected 10 ordinary one-round states and evaluated three
current candidate questions with both counterfactual answers through the
production generated-hypothesis, validity, and history-filtering pipeline.

## Result

- Candidate rows: `30` across `10` states.
- Mean within-state expected truth-coverage spread: `0.2286`.
- Median spread: `0.1304`; maximum spread: `0.6567`.
- Immediate EIG's mean coverage regret relative to the best candidate in its
  own pool: `0.1414`; the regret is at least `0.20` in `3/10` states.
- Spearman correlation of immediate EIG with expected truth coverage: `0.1446`.

For example, at the Giraffe state, immediate EIG chose `Does it live in the
water?` with zero expected truth coverage, while `Is it larger than a
medium-sized dog?` achieved `0.6567`. At the Tiger state, immediate EIG chose
`Is it a mammal?` with zero expected coverage, while `Is it larger than a dog?`
achieved `0.4967`.

The screen therefore clears the *mechanism-exists* gate: regeneration and
filtering can make future target coverage strongly candidate-dependent, and
the immediate EIG order does not faithfully capture that variation. It does
not yet establish an actionable policy, because truth coverage is unavailable
to a deployed policy. The next pre-result gate is whether a target-free
model-support-retention score ranks this outcome signal.

## Mechanics and Cost

- `google/gemma-4-26b-a4b-it`, explicit `thinking: false`.
- 5,252 total OpenRouter requests, no reasoning tokens, one length finish.
- Run cost: `$0.09905632` of the `$0.50` cap.
- Raw records: `results/nonmyopic/animals_coverage_dynamics/20260718_gemma26b_recovery/COVERAGE_PROBE.json`.

The prior two DeepSeek attempts are invalid pre-result executions and remain
documented in the preregistration and experiment ledger.
