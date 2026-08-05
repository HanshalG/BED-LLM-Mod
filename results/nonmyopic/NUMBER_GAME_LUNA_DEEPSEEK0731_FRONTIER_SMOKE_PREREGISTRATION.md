# Number Game Luna + DeepSeek 0731 Frontier Smoke Preregistration

Date frozen: 2026-08-05

## Purpose

Screen two newly available budget models under the exact Number Game planner
interface used for the existing Qwen3.7 Plus and DeepSeek V4 Flash comparison.
This is a serving and support-quality gate, not an efficacy experiment.

## Frozen Models And Interface

- `openai/gpt-5.6-luna`, model seed `1080501`;
- `deepseek/deepseek-v4-flash-0731`, model seed `1080502`;
- reasoning disabled, temperature `0.7`, strict executable-rule JSON schema;
- the unchanged ten linked histories: two initial, four one-observation, and
  four two-observation prompts;
- retained-rejuvenation support construction and unchanged parser.

Each model receives exactly ten accepted requests. The runs are independent;
neither model's result authorizes changing the other's prompts, parser, seed,
or thresholds.

## Gates

Each candidate passes only if all existing linked serving gates pass:

- exactly ten parsed responses, accepted requests, and HTTP attempts;
- zero retries, provider-error retries, reasoning tokens, and forced exits;
- both initial generated supports contain at least 16 valid hypotheses;
- all eight conditioned generated supports contain at least 8 valid
  hypotheses;
- every linked merged first support contains at least 8 hypotheses;
- every linked merged second support contains at least 4 hypotheses;
- run cost is no more than `$0.10`.

Report initial and conditioned valid-support means/minima, merged support
minima, token counts, exact measured cost, and strict-parser diagnostics.

## Decision Rule

A candidate that fails any gate is not advanced under this interface. Every
candidate that passes may enter a separately frozen paired downstream-risk
test on the existing canonical target and validation bank. Generic benchmark
scores or lower price cannot override this task-specific gate.

No paper claim, model replacement, or fully fresh source/control execution is
authorized by this smoke alone.

## Budget

The authenticated opening balance is `$32.458288549`. Account-wide spend is
capped at `$5.00` per Europe/London calendar day using the cumulative-usage
ledger at `results/nonmyopic/openrouter_daily_budget/2026-08-05.json`.
This two-model block reserves at most `$0.20`; authorization fails closed if
the day's remaining allowance is lower.
