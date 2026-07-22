# Gated Sensor LLM Branch-Policy Confirmation

Registered: 2026-07-22, after the exact depth qualification and interface smokes, before any formal policy endpoint.

## Motivation

The frozen zero-call qualification established a second, non-spatial task family with a large exact depth-two advantage: `+1.2459` entropy-AUC nats over exhaustive d1 (95% CI `[+1.2401, +1.2516]`). This confirmation tests the actual LLM-Modulo claim: whether an LLM supplies useful branch-policy search bias while an exact Bayesian verifier performs rollout scoring.

## Frozen Design

- Environment: 32 equiprobable five-bit fault codes; weak screens at accuracy `0.65`; precise tests at `0.95`; three overlapping test panels; panel activation consumes one round and has exactly zero immediate EIG.
- Model: `google/gemma-4-26b-a4b-it` through OpenRouter, non-thinking, temperature zero, maximum 2,048 output tokens.
- Trials: 30 paired trials, 8 rounds, seed `24092`, K6 branch policies, 10,000 paired bootstrap replicates, trial concurrency 32, OpenRouter concurrency 128.
- Machine-assigned root coverage: at horizon two, every currently legal activation action occupies one fixed root slot. The LLM chooses its precise follow-up. Remaining slots use distinct LLM-selected measurement roots with positive/negative follow-ups.
- At horizon one, duplicate or activation roots are deterministically replaced by unused legal measurement roots in fixed action order. This repair is belief-independent, preserves valid unique roots, and is logged.
- Strict validation permits one model retry. A failed-closed run may resume only by revalidating accepted cells under the identical config and preserving all rejected responses.
- Exact rollout scoring, posterior updates, action execution, observation sampling, and metrics make zero LLM calls.

## Arms

1. `strategy_eig`: exact h2 score over each LLM branch-policy cell.
2. `shared_d1`: immediate EIG over roots from an identically generated LLM cell at that arm's reached state.
3. `exhaustive_d1`: exact immediate EIG over every legal action.
4. `random_strategy`: exact h2 score over K6 policies from the same machine-assigned activation-root grammar, with random distinct measurement roots and random legal follow-ups.
5. `exhaustive_d2`: exact full-action h2 oracle.

Hidden targets and action-indexed observation uniforms are paired across all arms. Entropy AUC and truth-log-posterior AUC average the post-action values over all eight rounds.

## Gate

The primary endpoint is entropy AUC. The confirmation passes only if all mechanics checks pass and StrategyEIG has a strictly positive paired 95% bootstrap lower bound against each of:

- shared-cell d1;
- exhaustive d1;
- matched random branch policies.

Truth-log-AUC intervals corroborate but do not define the primary gate. The gap to exhaustive d2, final entropy, final MAP accuracy, selected actions, rejection rate, repairs, and cost are secondary.

## Cost

The formal run is projected at `$0.25` and has a hard per-run cap of `$0.50` within the existing `$40` project ledger. Exact scoring and all control arms are local.

## Smoke Disclosure

The first 2-trial, 3-round interface smoke failed closed after 11 requests because Gemma repeated a terminal precise root after one retry; it cost `$0.00598281`. This motivated the frozen deterministic terminal deduplication. A fresh smoke then passed with 10 accepted cells, two rejected responses corrected on retry, one terminal repair, and zero rollout-scoring LLM calls. It cost `$0.00564344`. Over only two descriptive pairs, StrategyEIG matched exhaustive d2 and gained `+0.3145` entropy-AUC nats over matched random; these values are not pooled with the formal run.

## Registered Outcome

The v1 confirmation failed closed and produced no policy endpoint. Across the initial run and 15 identical-config resumes, 385 cells were accepted and revalidated, but the cache then plateaued because temperature-zero Gemma deterministically repeated invalid cells. The audit preserved 662 rejected attempts: 378 illegal follow-up actions, 156 illegal root actions, 115 incorrect outcome-key sets, 11 invalid JSON responses, and two incomplete fences. Twenty-seven terminal roots were repaired under the registered deterministic rule. The 16 serving runs made 1,047 requests and cost `$0.76078769` in total.

No invalid policy was scored or executed, and no partial trial was used as an endpoint. This is a failed literal-action interface result, not a negative StrategyEIG performance result. The frozen failure is reported in `GATED_SENSOR_STRATEGY_V1_FAILURE_RESULT.md`; any replacement interface requires a new preregistration.

## Command

```bash
set -a; source .env; set +a
python scripts/nonmyopic_gated_sensor_strategy_prior.py \
  --config configs/config_nonmyopic_gated_sensor_openrouter.yaml \
  --run-id nonmyopic-gated-sensor-strategy-confirmation-20260722 \
  --output-dir results/nonmyopic/gated_sensor_strategy_confirmation_20260722 \
  --num-trials 30 --num-rounds 8 --num-strategies 6 \
  --seed 24092 --bootstrap-replicates 10000 --trial-concurrency 32
```
