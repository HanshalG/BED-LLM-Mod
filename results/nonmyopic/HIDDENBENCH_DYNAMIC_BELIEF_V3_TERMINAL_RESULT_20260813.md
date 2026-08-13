# HiddenBench Dynamic-Belief V3 Terminal Result

Date: 2026-08-13

Status: **serving schema/transport null; exact V3 cohort closed.**

## Result

The final four-row reserve cohort passed the frozen metadata-only source admission,
and the complete ten-call dynamic-belief implementation, independent verifier,
pass-token boundary, endpoint custodian, and account-wide budget wrapper were pushed
before execution. The exact transaction then stopped during its first four concurrent
root requests, before any answer-conditioned refresh, routing, label-free score, or
endpoint stage.

Two requests returned charged `stop` responses whose top-level envelopes matched the
strict root schema. The other two returned zero-cost `error` responses whose content
was the response-schema object itself (`name` plus `schema`), identically in both
cases. Strict parsing rejected those echoed objects with `root response has the wrong
fields`. No retry was permitted by the frozen protocol.

| Quantity | Result |
|---|---:|
| HTTP response records | 4 |
| charged `stop` responses | 2 |
| strict root envelopes | 2 |
| zero-cost `error` responses | 2 |
| identical response-schema echoes | 2 |
| refresh / router / auditor responses | 0 / 0 / 0 |
| label-free result / verification / pass token | 0 / 0 / 0 |
| registered-answer / endpoint artifacts | 0 / 0 |
| exact transaction cost | `$0.000668000` |

Authenticated cumulative credits/usage/balance closed at
`$245.000000000 / $220.179020166 / $24.820979834`. Conservative Aug-13
account-wide spend from the frozen `$220.134128880` boundary is `$0.044891286`,
below the hard `$5.00` cap.

## Interpretation

This is a serving schema/transport null, not evidence about HiddenBench semantic
calibration, answer-conditioned belief regeneration, depth-two planning, or policy
efficacy. The two syntactically usable roots are an incomplete, selected-by-transport
subset and are not scored. Registered answers and policy endpoints remained sealed.

The exact V3 protocol, four-row cohort, prompts, model, and seeds are closed and must
not be repaired or rerun. Together with V2 consuming four reserve rows and V3 consuming
four more, only one original reserve row remains, so the preregistered four-task
HiddenBench mechanics route cannot form another untouched cohort. HiddenBench is
therefore closed for this project cycle. A future benchmark/interface must first
demonstrate that its provider returns model content rather than schema echoes under
the exact structured-output route, before any task cohort is selected.

No development, confirmation, endpoint, or paper-efficacy claim is authorized.

## Integrity

- mechanics protocol SHA-256:
  `cab055aed50b22abfc1b90e720523882d8d6320616f0ab5f474a9340bf428bba`;
- source manifest SHA-256:
  `e69f9f3f6373f68d2164a04ea8f5f00fa773984fb46281f4e5dd5b58b86c77ca`;
- source audit SHA-256:
  `c639c4ad20b1e7d02d1f3f8bc9895ce64e08b6ea4beb6f1ce863b348d9bda87f`;
- execution binding SHA-256:
  `2809f954a1924fbcdd757d20f5cef9fbecf725eb0cd9dfd222f6a6053b652518`;
- raw partial bank SHA-256:
  `327ebb9bbba772ce929503687c0176db543d82969d13699025a9d32c347c4c9d`;
- run log SHA-256:
  `bfc5b90b3146f7f31fa8199cc73e95fb79c6e93b834ac846fbca7408a6def5bb`;
- failure SHA-256:
  `299173b7c0299239d32685a8541d6527b2fce1bbc304c50267e1997f8b34ca82`;
- reconciled ledger SHA-256:
  `c005ded5177bd862cd20c392bba37af2f03a95f099b5eae543ce707a34bdb771`;
- independent terminal audit SHA-256:
  `95d25e8c55b917d2e9caf42026a30d29cff4d34f365f522fab57fdf3fc6b38a4`.
