# RegretBench DeepSeek Support-Recovery Implementation

Date: 2026-08-07

**Status: ready and unopened. Model calls: 0. Cost: $0.**

## Implemented Contract

The implementation binds the official RegretBench source audit, frozen
mechanics/development splits, strict eight-hypothesis/four-question schema,
official post-selection semantic mapper, exact cooperative environment answer,
same-seed adjacent conditioned/blind refreshes, conservative lexical truth
coverage, paired bootstrap, and all preregistered mechanics/scientific gates.

Development now requires a validated passing exact-10 smoke artifact. It cannot
be invoked as a standalone efficacy run. Public artifacts omit generated
questions, answers, aliases, facets, raw responses, and true intent indexes;
those controls stay in `private/`.

The dated Aug 8 executor requires the already-frozen Luna naive baseline smoke
to pass first. It inherits that run's account-wide opening usage, reconciles
spend as the maximum of posted and locally measured cost after each stage, and
refuses the combined `$0.20 + $0.50` caps if they do not fit inside the same
Europe/London `$5.00` day.

## Verification

```text
pytest -q \
  tests/test_regretbench_deepseek_support_recovery.py \
  tests/test_regretbench_deepseek_support_recovery_daily.py \
  tests/test_regretbench_llm_native_source_audit.py \
  tests/test_openrouter_daily_budget.py \
  tests/test_validate_experiments_ledger.py

20 passed in 1.87s
```

The synthetic development fixture traverses the complete 64-task, 192-response
path through privacy checks, official question mapping, matched dispatch,
coverage scoring, bootstrap, gates, and public/private serialization. It is an
instrument test only and supplies no scientific evidence.

## Hashes

- source audit runner:
  `6906a6db52f26fb781c59b76d29ff095adaef0d85154c6b28d8a396402028163`
- support-recovery runner:
  `7c6aec662b564085c6670a8ac3e9cec47fa63370e1c72ae68d5cee791bfac849`
- Aug 8 daily executor:
  `3c0c90e863160b35772aefea2afdb046edd4555ea009fff9041bc766514a48dc`
- source preregistration:
  `7d68263cb75e120bccf69a892ed08ec459343b44cf51ded61b240398e0fd0b5c`
- support-recovery preregistration:
  `32825b97d9c73bbe940d2f0dfff5cb5deebd80f2cfec16812f227659a88a6dfe`
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
