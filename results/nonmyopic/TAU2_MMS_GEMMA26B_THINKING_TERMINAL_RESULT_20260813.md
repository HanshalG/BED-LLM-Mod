# Tau2 MMS Gemma 26B Thinking Terminal Result

Date: 2026-08-13

## Disposition

The prospectively frozen `google/gemma-4-26b-a4b-it` thinking calibration
failed closed at the serving gate. It authorizes nothing. No repair, task
success, policy-development, or confirmation endpoint was opened.

The exact model/interface/cohort/seeds are closed and must not be rerun.

## Frozen Execution

- Fresh reserve positions 18--20 in each MMS family, six episodes total.
- Twelve seeded structured requests, concurrency two, zero retries.
- Thinking budget 8192 tokens plus a 512-token final-answer allowance inside
  the exact 8704-token request cap.
- Forced-final continuation disabled.
- Full response bank written before any official calibration observation.
- Account-wide stage cap `$0.12`; actual local cost `$0.01758888`.

All 12 requests and all 12 HTTP attempts completed. The run used 6,476 prompt
tokens, 49,237 completion tokens, and 41,720 reasoning tokens. Nine requests
stopped normally. Three exhausted the 8,704-token cap with `finish_reason`
`length` and yielded empty final answers. Strict parsing therefore failed on
the first empty root answer.

## Label-Free Serving Audit

The complete bank and every payload identity pass the independent verifier's
pre-label checks. Among the final answers:

- root requests: 5/6 exact JSON schemas, one empty length exit;
- native requests: 4/6 exact JSON schemas, two empty length exits;
- total: 9/12 exact schemas and 3/12 empty length exits.

`ORDERING.json` records
`official_calibration_loaded_after_complete_bank=false`. Consequently no
source-label comparison, Brier score, posterior score, rank fidelity,
depth-two decision, or semantic pass/fail was computed. This is serving
evidence only, not evidence about semantic calibration or policy efficacy.

## Budget And Provenance

The frozen Aug 13 account-wide opening usage was `$220.134128880`. Closing
authenticated usage was `$220.162584176`; conservative account-wide spend was
therefore `$0.03144790600001102`, well below the hard `$5.00` cap.

- execution binding: `21c2725ea6f173df2bafeaf7885060a839771b8ed30dc7841335c5b4c585ae23`
- partial bank: `5e41fd4e7874de95ebc964ea6f8435f4737fd3ac03c048c5be81012cc4d8d699`
- raw bank: `50e51b32119a48e2606ccda51de16b69c92ef3b54cd8099ae059d46f2d321dc9`
- failure: `497cf0c630dfa505496a1390d7008383d1e77a63addbc80e31df104d068879a5`
- ledger: `f5bb1145e834448eba12bfc1881770ca8cad1ec0af917840e2d133adba9edac2`
- run log: `74269109df4d49119dd6fd3a0c63f5dad669b193981541e38ed623503ab99efa`

## Next Research Constraint

Do not repair or rerun this cohort. A successor must be prospectively frozen
on untouched reserve episodes and new seeds. It must address reasoning
exhaustion as a serving property before labels, for example with a compact
thinking cap that leaves a guaranteed final-answer budget or a separately
frozen continuation contract. It must retain the same semantic gates and
cannot use these unopened labels to tune the interface.
