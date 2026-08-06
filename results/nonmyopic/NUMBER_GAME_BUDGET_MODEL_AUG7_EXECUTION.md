# Number Game August 7 Budget-Model Execution

Date frozen: 2026-08-06, before the August 7 control or any budget-model gate
seed was opened.

## Sequence

Run these components sequentially on 2026-08-07 Europe/London. Do not launch
the two model gates concurrently: they share one account-wide daily ledger and
must reconcile locally measured cost between components.

### 1. Sealed Qwen Control

```bash
set -a
source .env
set +a

/Users/hanshalgoyal/.conda/envs/20_questions_env/bin/python \
  scripts/number_game_qwen_fully_fresh_control_daily_execute.py \
  --run-dir results/nonmyopic/number_game_qwen_fully_fresh_daily_stages/number-game-qwen-fully-fresh-daily-stages-20260806T000200Z \
  --run-id number-game-qwen-fully-fresh-daily-stages-20260806T000200Z \
  --daily-ledger results/nonmyopic/openrouter_daily_budget/2026-08-07.json
```

Continue only when `CONTROL_DAILY_EXECUTION.json` says `complete_verified`.
The executor then authorizes the exact tail entries supported by the measured
remaining allowance.

### 2. GPT-5.6 Luna Reliability128

```bash
/Users/hanshalgoyal/.conda/envs/20_questions_env/bin/python \
  scripts/number_game_budget_model_reliability128.py \
  --model openai/gpt-5.6-luna \
  --output-dir results/nonmyopic/number_game_budget_model_reliability128/number-game-budget-model-reliability128-luna-20260807 \
  --run-id number-game-budget-model-reliability128-luna-20260807 \
  --daily-ledger results/nonmyopic/openrouter_daily_budget/2026-08-07.json \
  --qwen-control-result results/nonmyopic/number_game_qwen_fully_fresh_daily_stages/number-game-qwen-fully-fresh-daily-stages-20260806T000200Z/RESULT.json
```

### 3. DeepSeek V4 Flash 0731 Reliability128

```bash
/Users/hanshalgoyal/.conda/envs/20_questions_env/bin/python \
  scripts/number_game_budget_model_reliability128.py \
  --model deepseek/deepseek-v4-flash-0731 \
  --output-dir results/nonmyopic/number_game_budget_model_reliability128/number-game-budget-model-reliability128-deepseek0731-20260807 \
  --run-id number-game-budget-model-reliability128-deepseek0731-20260807 \
  --daily-ledger results/nonmyopic/openrouter_daily_budget/2026-08-07.json \
  --qwen-control-result results/nonmyopic/number_game_qwen_fully_fresh_daily_stages/number-game-qwen-fully-fresh-daily-stages-20260806T000200Z/RESULT.json
```

Each reliability gate has a `$0.10` hard cap. A `gated_null` is a valid banked
result and must not be rerun or repaired.

### 4. Selected-Model Stress3584

Run only when the control ledger contains the waiting stress entry. The runner
itself selects the model from both frozen gate artifacts. The artifact argument
may be `RESULT.json` or `FAILURE.json` for a failed-closed gate.

```bash
/Users/hanshalgoyal/.conda/envs/20_questions_env/bin/python \
  scripts/number_game_budget_model_stress3584.py \
  --output-dir results/nonmyopic/number_game_budget_model_stress3584/number-game-budget-model-stress3584-20260807 \
  --run-id number-game-budget-model-stress3584-20260807 \
  --daily-ledger results/nonmyopic/openrouter_daily_budget/2026-08-07.json \
  --luna-result results/nonmyopic/number_game_budget_model_reliability128/number-game-budget-model-reliability128-luna-20260807/RESULT.json \
  --deepseek-result results/nonmyopic/number_game_budget_model_reliability128/number-game-budget-model-reliability128-deepseek0731-20260807/RESULT.json
```

If neither model passes reliability128, stress3584 makes zero calls and banks
`no_eligible_model`. If the exact remaining daily allowance is below `$1.55`,
the waiting stress entry is absent and the runner refuses before model calls.

## Budget Envelope

- Qwen control: expected about `$3.21`, hard cap `$4.25`;
- both reliability gates: expected about `$0.09`, combined hard cap `$0.20`;
- stress3584: expected about `$1.2--$1.5`, hard cap `$1.55`;
- expected day: about `$4.4--$4.8`;
- account-wide day: hard `$5.00`, enforced before every component.

Every successful or failed paid component checkpoints the larger of locally
measured and posted account spend. No policy endpoint is used to choose the
budget model or decide whether the stress gate runs.
