# Tau2 MMS Gemma 26B Budgeted-Thinking Terminal Result

Date: 2026-08-13

## Disposition

The prospectively frozen `google/gemma-4-26b-a4b-it` budgeted-thinking
calibration failed closed at the serving gate. It authorizes nothing. No
repair, task-success, policy-development, or confirmation endpoint was opened.

The exact model/interface/cohort/seeds are closed and must not be rerun.

## Frozen Execution

- Fresh availability-bound reserve cohort: four `mms_abroad` episodes at
  positions 21--24 and two `mms_home` episodes at positions 21--22.
- Twelve seeded structured requests, concurrency two, zero retries.
- Explicit provider reasoning cap 4,096 tokens and total completion cap 4,608,
  intended to leave 512 final-answer tokens.
- Forced-final continuation disabled.
- Per-request finish, reasoning-token, and content evidence banked before any
  parse or official calibration observation.
- Account-wide stage cap `$0.08`; actual local cost `$0.01277538`.

All 12 requests and all 12 HTTP attempts completed. The run used 6,558 prompt
tokens, 32,929 completion tokens, and 28,325 reported reasoning tokens. Nine
requests stopped normally. Three exhausted the 4,608-token total cap, returned
`finish_reason=length`, and yielded empty final answers.

## Serving Result

The requested 4,096-token reasoning maximum was not a hard serving guarantee
on this route. The three length exits reported 4,131, 4,185, and 4,117 reasoning
tokens. Therefore all of these frozen gates failed:

- every request reasoning count at most 4,096;
- every request finishes with `stop`;
- every response has nonempty final content;
- zero forced thinking exits.

The remaining serving conditions passed: exact 12 accepted requests and HTTP
attempts, zero retries/provider-error retries, positive aggregate reasoning,
zero forced-final requests/successes, exact attempt identity, and cost below the
stage cap.

## Label-Free Codec Audit

The complete raw bank exists. Without loading official observations, strict
codec parsing finds:

- root requests: 4/6 exact schemas and two empty length exits;
- native requests: 5/6 exact schemas and one empty length exit;
- total: 9/12 exact schemas and three empty length exits.

`ORDERING.json` records
`official_calibration_loaded_after_complete_bank=false`. Consequently no source
label comparison, Brier score, posterior score, rank fidelity, depth-two
decision, or semantic pass/fail was computed. This is serving evidence only,
not evidence about semantic calibration or policy efficacy.

## Budget And Provenance

The frozen Aug 13 account-wide opening usage was `$220.134128880`. Closing
authenticated usage was `$220.176749606`; the chained conservative spend is
`$0.04422328600000469`, below the hard `$5.00` cap.

- execution binding: `5c80f6c92230de80b9682a96dd949bd2a6c81bceb8aee2c01409211a14c51a70`
- partial bank: `fbb0ee231e4e11c9fd89b8c97e349ae11425a2f629a4f85201f1242f302a3746`
- raw bank: `5a38ed529079d9560b1e228483541c347c978f7ac72595069562c0c485b7bf15`
- serving bank: `cb96557382a47238cf76e255ff224ee83b9093d91938afab45c1a1c342a99963`
- ordering bank: `1377e4f45e3a66a9a109d57e672bb0c25544c1954f7bf1209d5e9fab0fc8d6a1`
- failure: `634c741d1b83fc365451dc959977bb27517d47dc6a9dd6e50fdd074e193b6aed`
- ledger: `64a5c5485d4fb9ead5a75890907a10c4dca246bb4408c38cf70323c4276da89d`
- run log: `9748472efc85a23c514e71632415b562b16db859def4a103894057fc0ca31eff`

## Research Consequence

Explicit OpenRouter `reasoning.max_tokens` reduced cost and average reasoning
but did not reserve a reliable final-answer channel for this Gemma route. Do
not repair or rerun this cohort. A future interface must pass a fresh serving
gate and cannot infer semantic efficacy from these unopened labels. Since the
mixed-family MMS reserve is now exhausted, continuing this exact Tau2 line
would also require a prospectively justified new source cohort rather than
relaxing family coverage.
