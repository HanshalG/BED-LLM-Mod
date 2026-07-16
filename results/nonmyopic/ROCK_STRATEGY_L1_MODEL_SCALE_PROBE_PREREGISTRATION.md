# Rock Strategy-Prior L1 Model-Scale Probe

Registered on 2026-07-16 before this probe is launched. This is the single final
probe authorized in `STATE.md`; terminal consolidation follows either outcome.

## Question

The completed 26B non-thinking Rock attempts produced no valid endpoint because of
strategy-grammar failure, while the continuous L3 endpoint did not establish the LLM
prior against the random-plan control. This bounded probe tests the one open capability
objection: can a materially stronger reasoning-capable generator supply a sufficiently
useful compact strategy prior on the exact short-horizon Rock anchor?

## Frozen Difference

Only the strategy-generation model changes from the frozen L1 anchor:

- Questioner / strategy generator: `google/gemma-4-31b-it`, OpenRouter reasoning
  enabled, temperature `0`, 1,024 reasoning-token budget plus 256 final-output tokens.
- The L1 runner does not invoke the configured answerer. It remains the baseline 26B
  non-thinking spec solely because the shared config schema requires a pair.

The live OpenRouter catalog queried at registration lists reasoning support, a 262,144
token context, `$0.22/M` prompt-token pricing, and `$0.55/M` completion-token pricing
for this model. It is the strongest adapter-compatible reasoning model fitting the
authorized `$1–2` run envelope; more expensive frontier models do not fit that cap at
the L1 request volume.

## Frozen L1 Configuration

- Maps: Rock Diagnosis Figure 4 `3-6` and `5-7`.
- Fresh keyed seed: `12034`; 30 paired trajectories per map; 8 rounds each.
- K=4 compact reactive strategies; horizon 2; trial concurrency 32.
- Strict current Rock grammar and prompt, including remote-check clarification and
  current-root move/check validation; at most one validation-feedback retry; fail
  closed on a remaining invalid cell.
- Exact dynamics, full-vector posterior, likelihood, score, observation schedule,
  decode, scorer-unit accounting, and random-strategy sampler are unchanged.
- Arms remain StrategyEIG, exhaustive d2, shared d1, equal-call/equal-compute width,
  and grammar-matched random strategies. The stronger model is used for the LLM cells;
  the exact scorer makes zero LLM calls.

## Endpoints And Decision

The model-scale capability gate is the `STATE.md` rule: on **both maps**, the paired
final entropy gain of StrategyEIG has a strictly positive 10,000-replicate bootstrap
lower endpoint versus shared d1 and versus random strategies. Width and exhaustive-d2
comparisons retain their original reporting role, and the stricter all-three-control L1
intersection is reported separately; neither may be omitted or replaced by a secondary
metric.

If the capability gate passes, report a frontier-capability threshold result, while
also stating whether the stricter original all-control gate passed. If it fails or the
runner fails closed, conclude that the negative result is robust to this single model
scale probe. There will be no further model, grammar, environment, or horizon rerun.

## Budget And Execution

The run uses `configs/config_nonmyopic_rock_strategy_l1_gemma31b_thinking_openrouter.yaml`.
The adapter's projected-cost guard is `$1.50`; the hard per-run cap is `$2.00`. A
conservative bound from the prior L1 prompt volume (about 1.19M prompt tokens), 1,440
logical cells, and the configured maximum 1,280 completion tokens per cell is about
`$1.28` before the small bounded-retry allowance. The OpenRouter spend ledger has
`$21.43` remaining of the `$40` project authorization at registration.

No smoke, model-selection pilot, or policy endpoint is run before this one formal
attempt. The run will use `source .env` to obtain the non-committed OpenRouter key and
will log all physical requests, raw rejects, retries, reasoning tokens, and cost.
