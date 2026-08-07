# RegretBench DeepSeek Dynamic Depth-Two Policy Implementation

Date: 2026-08-07

**Status: ready and unopened. Model calls: 0. Cost: $0.**

The formal development result is independently reconstructed from all raw
belief, branch, selection, truth-control, and realized-history artifacts by
`scripts/regretbench_deepseek_result_verify.py`. The daily executor accepts the
producer result only when this zero-call replay matches its primary endpoint,
statistics, mechanics gates, and status.

The distinct-action amendment additionally requires the second question to map
to a different official semantic facet from the first. Rephrasing the same
facet cannot create a valid two-step trajectory: all three enriched-smoke paths
must be novel, and every primary policy must clear `40/64` novel second actions.
The optional Luna baseline uses the same target only as an availability
diagnostic and remains unable to affect primary status.

The valid-trajectory endpoint amendment prevents invalid dialogue from earning
scientific credit. Unsupported first/second actions or repeated second facets
receive zero scored truth mass, Brier `1`, and floor log loss. Their raw
regenerated masses remain explicitly descriptive, while every valid trajectory
retains the original aligned likelihood endpoint unchanged.

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
  the shared fresh-regeneration endpoint but unable to gate or abort primary
  execution; and
- exact official RegretBench execution followed by an aligned generated-
  likelihood primary endpoint and fresh-regeneration secondary endpoint.

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
maximum primary DeepSeek requests                    8,768
naive first/final endpoint supports                    128
maximum DeepSeek requests                            8,896
naive Luna questions                                   128
maximum combined requests                            9,024
```

Every conditioned response is immediately followed by its prompt-only mate in
the same batch with the same requested seed. All four simulated roots also
share each task/hypothesis/draw seed. Realized roots share task-level first and
final refresh seeds. Formal concurrency is `128`. Every enriched support must
contain exactly eight distinct particles; duplicate particles fail instead of
silently changing rollout width.

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

The dynamic policy makes zero calls after any scientific predecessor null or
enriched-policy smoke failure. A naive smoke failure disables only the
descriptive baseline. The primary DeepSeek, optional Luna, and optional naive
endpoint DeepSeek adapters share one run ID, so the hard OpenRouter tracker
reserves every concurrent request against one combined `$3.50` cap. A formal
baseline failure is banked without changing primary mechanics or science.
Confirmation remains sealed regardless of a development null.

The daily preflight now validates both live model records before adapter
construction. DeepSeek must retain seeded structured output and sufficient
price-adjusted `$0.0015` request coverage; Luna must retain multimodal,
reasoning, structured-output, and `$0.008` request coverage. DeepSeek failure
stops the primary protocol; Luna failure records `unavailable_preflight`, skips
all naive calls, and continues primary execution. This is a fail-before-
dispatch operational amendment only.

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

76 passed in 15.79s
```

The synthetic full run materializes all `8,256` planning responses and every
selected realized path, all `128` Luna questions, and all `128` DeepSeek
naive-path endpoint supports, then traverses parsing, scoring, official
mapping, endpoint computation, bootstrap, privacy, request accounting, and
public/private serialization. Separate exact-scale adversarial rehearsals prove
that a formal Luna failure and a smoke-disabled baseline both leave primary
mechanics and science computation intact. Their outcomes are instrument
fixtures, not evidence. A seed-only adversary additionally proves that the old
root-specific schedule could create a spurious `0.135796` candidate-risk
spread, while the bound task-level CRN schedule makes it exactly zero. A
hand-built aligned endpoint gives exact mass `.25`/Brier `.5625`; an adversarial
fresh-endpoint reversal is reported but cannot change primary gates.

## Bindings

- preregistration:
  `03c0f5bd48bd021d696e8641c29942644306b2fd3532157c52af3fc01b6d0f54`
- policy core:
  `ca75b5031d7d680975df894d905b605b9445436288f64a4f3cdfa15025e6da03`
- policy daily executor:
  `ac6b8d58f8cc442ef1f5d3ed56c2cdb3caf5a46f919b3bfe372c7a415e62179c`
- amended support-recovery daily executor:
  `0ee8dbfb632b64ca5bb71fe23495e74f00b7fb46fef27559342af3fd47a08303`

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
