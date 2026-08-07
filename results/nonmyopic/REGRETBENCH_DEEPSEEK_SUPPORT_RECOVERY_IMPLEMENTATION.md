# RegretBench DeepSeek Support-Recovery Implementation

Date: 2026-08-07

**Status: ready and unopened. Model calls: 0. Cost: $0.**

## Implemented Contract

The implementation binds the official RegretBench source audit, frozen
mechanics/development splits, strict eight-hypothesis/four-question schema,
official post-selection semantic mapper, exact cooperative environment answer,
same-seed adjacent conditioned/blind refreshes, conservative lexical truth
coverage, paired bootstrap, and all preregistered mechanics/scientific gates.
The matcher explicitly rejects unequal equal-length strings, and the dated
executor binds the exact matcher-bearing core hash before any request.

Development now requires a validated passing exact-10 smoke artifact. It cannot
be invoked as a standalone efficacy run. Public artifacts omit generated
questions, answers, aliases, facets, raw responses, and true intent indexes;
those controls stay in `private/`.

The dated Aug 8 executor requires the already-frozen Luna naive baseline smoke
to pass first. It inherits that run's account-wide opening usage, reconciles
spend as the maximum of posted and locally measured cost after each stage, and
refuses the combined `$0.20 + $0.50` caps if they do not fit inside the same
Europe/London `$5.00` day.

Before any adapter construction, the daily preflight now also requires the
exact live 0731 endpoint, seeded structured output, frozen context/completion
limits, and enough price-adjusted request reservation for `2,200` output plus
at least `4,096` prompt tokens. The frozen exhaustive serialized-request
envelope is `3,354` bytes; live coverage at audit time is `12,266.67` tokens.

## Verification

```text
pytest -q \
  tests/test_regretbench_deepseek_support_recovery.py \
  tests/test_regretbench_deepseek_support_recovery_daily.py \
  tests/test_regretbench_llm_native_source_audit.py \
  tests/test_openrouter_daily_budget.py \
  tests/test_validate_experiments_ledger.py

25 passed in 1.89s
```

The synthetic development fixture traverses the complete 64-task, 192-response
path through privacy checks, official question mapping, matched dispatch,
coverage scoring, bootstrap, gates, and public/private serialization. It is an
instrument test only and supplies no scientific evidence.

## Hashes

- source audit runner:
  `6906a6db52f26fb781c59b76d29ff095adaef0d85154c6b28d8a396402028163`
- support-recovery runner:
  `7e227e4d3a125b817dd45c31ce6b1fc94c24bae6ee982ce59f2a9082065752c2`
- Aug 8 daily executor:
  `0ee8dbfb632b64ca5bb71fe23495e74f00b7fb46fef27559342af3fd47a08303`
- source preregistration:
  `7d68263cb75e120bccf69a892ed08ec459343b44cf51ded61b240398e0fd0b5c`
- support-recovery preregistration:
  `666dfde7a49d512a23c5cc8bd78a6e3caa5386f71ce41c07c4c5c123aaab96d2`
- source audit result:
  `d7a10f15ecf6779520fbb20712c8d43059d8b03168fa904fe72e82ffe87578de`
- source protocol manifest:
  `8a46b40395487aae0857d1a61d6f680beef579414c4580acf09d7e0b30b38e97`

## Execution

After the Luna baseline smoke has passed on Aug 8:

```bash
source .env
/opt/anaconda3/bin/python \
  scripts/regretbench_deepseek_support_recovery_daily.py --preflight
/opt/anaconda3/bin/python \
  scripts/regretbench_deepseek_support_recovery_daily.py
```

The second command is allowed only when the read-only preflight reports
`ready_without_paid_calls`. A smoke stop or development null is banked exactly;
there is no response repair, threshold change, rerun, or confirmation access.
