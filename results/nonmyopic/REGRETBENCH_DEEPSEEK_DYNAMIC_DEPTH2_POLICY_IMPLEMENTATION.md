# RegretBench DeepSeek Dynamic Depth-Two Policy Implementation

Date: 2026-08-07

**Status: ready and unopened. Model calls: 0. Cost: $0.**

## What Is Ready

The implementation constructs a genuinely LLM-native sequential BED tree:

- eight generated semantic hypotheses and four generated questions per belief;
- four generated answer likelihoods per hypothesis;
- two answer-conditioned and two same-seed history-blind support draws for
  every root/hypothesis pair;
- dynamic terminal truth-group Brier after an EIG-selected follow-up;
- compute-matched myopic, fixed-support depth-two, matched history-blind, and
  random-root controls; and
- a separate Luna medium-reasoning naive-thinking trajectory, measured with
  the same DeepSeek truth-mass endpoint but excluded from scientific gates; and
- exact official RegretBench execution followed by a continuous final
  truth-mass Brier and log-loss endpoint.

Root selection is checkpointed before hidden truth access. Actual histories are
cached by selected root, so policies choosing the same root share the exact
environment replies, regenerated support, second question, and final support.
Generated questions, replies, aliases, facets, true intent indexes, and raw
responses are absent from public artifacts.

## Exact Accounting

```text
initial supports                                      64
conditioned/blind branches     64*4*8*2*2           8,192
planning total                                      8,256
realized first histories                      at most 256
realized final histories                      at most 256
naive first/final endpoint supports                    128
maximum DeepSeek requests                            8,896
naive Luna questions                                   128
maximum combined requests                            9,024
```

Every conditioned response is immediately followed by its prompt-only mate in
the same batch with the same requested seed. Formal concurrency is `128`.
Every enriched support must contain exactly eight distinct particles; duplicate
particles fail instead of silently changing rollout width.

## Budget Chain

The Aug 8 executor validates the exact support-recovery result and inherits its
original account-wide day opening and recorded spend. It rechecks the full
remaining stage cap immediately before enriched smoke, naive-thinking smoke,
and development.
The full chain's maximum exposure is `$4.80`:

- Luna naive smoke: `$0.20`;
- support-recovery smoke and development: `$0.70`;
- enriched policy smoke: `$0.20`;
- naive-thinking smoke: `$0.20`;
- mixed DeepSeek/Luna policy development: `$3.50`.

The dynamic policy makes zero calls after any predecessor null/failure or a
smoke failure. Both formal adapters share one run ID, so the hard OpenRouter
tracker reserves every concurrent request against one combined `$3.50` cap.
Confirmation remains sealed regardless of a development null.

## Verification

```text
pytest -q \
  tests/test_regretbench_deepseek_dynamic_depth2_policy.py \
  tests/test_regretbench_deepseek_dynamic_depth2_policy_daily.py \
  tests/test_regretbench_deepseek_support_recovery.py \
  tests/test_regretbench_deepseek_support_recovery_daily.py \
  tests/test_regretbench_llm_native_source_audit.py \
  tests/test_bongard_openworld_luna_naive_first_link.py \
  tests/test_openrouter_model.py

61 passed in 7.85s
```

The synthetic full run materializes all `8,256` planning responses and every
selected realized path, all `128` Luna questions, and all `128` DeepSeek
naive-path endpoint supports, then traverses parsing, scoring, official
mapping, endpoint computation, bootstrap, privacy, request accounting, and
public/private serialization. Its outcomes are instrument fixtures, not
evidence.

## Bindings

- preregistration:
  `c1a19408c6cea5bb32023a7f9f003afac3a210d19fae0842a47a873f29b86b97`
- policy core:
  `fe611671fdf668733037b000d2574bec458b7ae6ee9f2d0a062f2c31d2fc3a55`
- policy daily executor:
  `05e76d125ae3d23c104b39338e39536f9d788fcdf310bc91757c446925965977`
- amended support-recovery daily executor:
  `ddd3ecbb9e8a559f6bd1dae2d6f85dc3835e8187ac0cb032e06a823925b529f4`

## Conditional Execution

Only after the support-recovery daily result is a literal clean pass:

```bash
source .env
/opt/anaconda3/bin/python \
  scripts/regretbench_deepseek_dynamic_depth2_policy_daily.py --preflight
/opt/anaconda3/bin/python \
  scripts/regretbench_deepseek_dynamic_depth2_policy_daily.py
```

The execution command is allowed only after `ready_without_paid_calls`. A pass,
null, mechanics failure, or transport failure is banked exactly once.
