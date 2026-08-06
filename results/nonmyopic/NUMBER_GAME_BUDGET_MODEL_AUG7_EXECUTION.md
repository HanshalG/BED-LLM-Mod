# Number Game August 7 Budget-Model Execution

Date frozen: 2026-08-06, before the August 7 control or any budget-model gate
seed was opened.

## Authoritative Command

Run exactly this command on 2026-08-07 Europe/London:

```bash
set -a
source .env
set +a

/Users/hanshalgoyal/.conda/envs/20_questions_env/bin/python \
  scripts/number_game_budget_model_aug7_execute.py
```

Do not launch the component CLIs separately. The orchestrator is the formal
execution path and runs components sequentially against the same account-wide
ledger:

1. sealed Qwen history-blind control and independent verification;
2. zero-call claim classification and immutable JSON/Markdown report;
3. GPT-5.6 Luna reliability128;
4. DeepSeek V4 Flash 0731 reliability128; and
5. the selected model's stress3584 when it was guaranteed after control or
   when the ledger-only post-reliability reconsideration finds its full
   `$1.55` cap still available.

The executor checkpoints after each component. A banked `RESULT.json` or
`FAILURE.json` is validated and never called again on resume. One model's
failed-closed reliability gate does not suppress the other independent gate;
the stress selector can use the surviving model. A preflight failure that
produces no component artifact stops the sequence. If neither model passes,
stress3584 makes zero calls and banks `no_eligible_model`.

The public sequence summary is written to
`results/nonmyopic/number_game_budget_model_aug7_execution/number-game-budget-model-aug7-20260807/RESULT.json`.
It records component artifact hashes, final model selection, and reconciled
account-wide spend without copying private responses.

The read-only launch check is:

```bash
set -a
source .env
set +a

/Users/hanshalgoyal/.conda/envs/20_questions_env/bin/python \
  scripts/number_game_budget_model_aug7_execute.py --preflight
```

It makes no model calls and writes no files. In addition to all frozen hashes,
paths, cases, and budget boundaries, it requires exact live Qwen, Luna, and
DeepSeek endpoints with text input, structured output, at least 4,200
completion tokens, and finite non-negative pricing.
The authoritative paid command automatically runs this complete check before
creating its wrapper directory or constructing any model adapter. A wrong-day
or failed fresh preflight leaves no artifact and cannot poison a later valid
launch. Resumption instead verifies banked components and never repeats them.

## Budget Envelope

- Qwen control: expected about `$3.21`, hard cap `$4.25`;
- both reliability gates: expected about `$0.09`, combined hard cap `$0.20`;
- stress3584: expected about `$1.2--$1.5`, hard cap `$1.55`;
- expected day: about `$4.4--$4.8`;
- account-wide day: hard `$5.00`, enforced before every component.

Every successful or failed paid component checkpoints the larger of locally
measured and posted account spend. No policy endpoint is used to choose the
budget model or decide whether the stress gate runs.

The original post-control rule still guarantees stress when the control costs
at most `$3.25`. If the control is slightly above that threshold, both fixed
reliability gates run first. Once both are terminal and independently
verified, the orchestrator may append the exact stress authorization when
reconciled recorded spend is at most `$3.45`. This one-time decision reads
only ledger identities, terminal statuses, and spend; it is forbidden from
reading model eligibility, support, parse, Brier, policy, or target values.
The amendment is frozen in
`NUMBER_GAME_BUDGET_MODEL_DEFERRED_STRESS_AUTHORIZATION_AMENDMENT.md`.

## Post-Control Claim Scope

The orchestrator automatically runs the zero-call claim classifier immediately
after independent control verification and before any reliability or stress
call. Classification failure stops the paid tail while preserving the banked
control; resume retries classification without repeating the control. A
completed-wrapper replay also re-runs the classifier idempotently, so a missing
or altered report fails closed.

The standalone command is retained only for explicit audit or recovery:

```bash
/Users/hanshalgoyal/.conda/envs/20_questions_env/bin/python \
  scripts/number_game_qwen_fully_fresh_claim_report.py \
  --run-dir results/nonmyopic/number_game_qwen_fully_fresh_daily_stages/number-game-qwen-fully-fresh-daily-stages-20260806T000200Z
```

It independently replays the public control, requires exact equality with the
banked verification, and freezes policy, dynamic-endpoint, and matched-control
claim families separately. The source hash already fixes the full fresh tier
as unreachable: a control pass can reach only
`nonmyopic_policy_with_partial_llm_mechanism`. This report does not change the
separately preregistered August 8--9 confirmation schedule.
