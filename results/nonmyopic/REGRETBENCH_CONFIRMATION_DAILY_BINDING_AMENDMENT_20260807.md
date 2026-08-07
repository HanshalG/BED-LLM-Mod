# RegretBench Confirmation Daily-Executor Binding Amendment

Date frozen: 2026-08-07

This final pre-response amendment binds the dated, fail-closed confirmation
executor required by the parent protocol and execution-binding amendment.

- parent protocol manifest:
  `7a782f02eb8c3b16d5b229cca309d02bce64df6432c5090c977d3e26d1f46498`;
- execution bindings:
  `a2537a081c2137082a3814b2913446fbbad854ff4c59dcd004edfe2a750ed34e`;
- confirmation producer:
  `7cbe10ec1dde5406d21dfb2ee02431ca5771e5e760c8bcb939f2cea94ae129d0`;
- independent confirmation result verifier:
  `f8a88ba92a079f2993838ea50cac5fb78546ba66f1428a173927b57c72eaa6aa`;
- dated executor
  `scripts/regretbench_deepseek_dynamic_depth2_confirmation_daily.py`:
  `f3b83377d4dda5571dddb1166b929d06d8fd6115607a566c67cf11233ba20eda`.

The executor is restricted to 2026-08-09 Europe/London. It derives the day's
usage boundary from the reconciled Aug 8 ledger, so delayed or unrelated usage
after that close counts against the `$5` cap. It reserves the full `$3.50`
confirmation cap before dispatch and reconciles spend as the maximum of posted
usage and locally measured accepted-request cost.

It opens calls only from a literal `passed` development result with mechanics
and science both passing, an independent `verified` raw-artifact replay, exact
result/verification/daily/ledger hashes, and a live DeepSeek0731 seeded strict
structured-output catalog contract. It requires pristine output paths and
makes no optional Luna calls.

After execution, the independent verifier must reconstruct the entire result.
A verified `gated_null` is banked as complete and authorizes nothing; it is not
retried. Any partial, hash drift, replay failure, date error, budget error, or
transport failure is failed closed.

This amendment changes no scientific field and makes zero model calls at zero
cost.
