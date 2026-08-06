# Fully Fresh Number Game Claim Decision Plan

Frozen: 2026-08-06, after the source result and before any fresh
history-blind control response.

## Purpose

The source already establishes a fresh depth-three policy gain over myopic EIG,
but it fails the frozen dynamic-versus-fixed endpoint family. Tomorrow's
matched control can still establish a fresh answer-conditioning mechanism; it
cannot retroactively turn the source into a full policy-mechanism replication.

The zero-call post-control classifier therefore keeps three gate families
separate:

1. **Non-myopic policy:** all source mechanics and all three depth-three versus
   myopic gates.
2. **Dynamic-support endpoint:** all source mechanics and all four dynamic
   versus compute-matched fixed-support gates.
3. **Matched conditioning:** all source and control mechanics and all four
   conditional versus history-blind control gates.

## Frozen Tiers

- `full_fresh_llm_native_replication`: all three families pass.
- `nonmyopic_policy_with_partial_llm_mechanism`: policy passes and exactly one
  or both mechanism families pass, but the full conjunction fails.
- `nonmyopic_policy_without_fresh_mechanism`: only the policy family passes.
- `fresh_mechanism_without_policy`: at least one mechanism family passes but
  the policy family fails.
- `fresh_replication_null`: neither policy nor mechanism evidence passes its
  complete frozen family.

Only the first tier authorizes a full fresh claim upgrade. Because the source
dynamic-support endpoint family has already failed, that tier is now
structurally unreachable for this cohort. A control-positive result can reach
only the partial tier and must be described as separate fresh policy and
matched-mechanism evidence.

This ceiling is executable and hash-bound to source `RESULT.json` SHA-256
`13fd3361a8ef8f525f68733182a9bdb30151e37a4a5e14dfb35dde78540ab523`:

```bash
/Users/hanshalgoyal/.conda/envs/20_questions_env/bin/python \
  scripts/number_game_qwen_fully_fresh_claim_report.py --preflight
```

It reports `claim_ceiling_frozen_without_control`, policy-family pass,
dynamic-endpoint-family fail, and maximum reachable tier
`nonmyopic_policy_with_partial_llm_mechanism`, with zero calls or files.

## Execution

After the paid control and independent verifier complete:

```bash
/Users/hanshalgoyal/.conda/envs/20_questions_env/bin/python \
  scripts/number_game_qwen_fully_fresh_claim_report.py \
  --run-dir results/nonmyopic/number_game_qwen_fully_fresh_daily_stages/number-game-qwen-fully-fresh-daily-stages-20260806T000200Z
```

The classifier independently replays the public control artifacts in memory,
requires exact equality with `CONTROL_VERIFICATION.json`, rejects non-finite or
inconsistent metrics, and banks `CLAIM_REPORT.json` plus `CLAIM_REPORT.md`
once. It makes no model calls.

This classification does not alter or authorize the separately preregistered
August 8--9 diversity-bonus confirmation. That experiment remains governed by
its own frozen seeds, gates, and daily execution wrappers.
