# Qwen 397B Successor Serving-Gate Failure

The first registered successor smoke used `qwen/qwen3.5-397b-a17b` with reasoning
enabled and a 512-token reasoning plus 256-token final-output budget. It failed closed
on the first of ten no-repair Rock L1 cells, before any policy endpoint was evaluated.

The provider reported 768 completion tokens, including 738 reasoning tokens, and a
`length` finish reason. The returned final content did not parse as JSON. The smoke
therefore failed the required zero-forced-exit / parseability gate after one request at
`$0.00405675`. This is a serving-budget failure, not evidence for or against the
StrategyEIG policy claim. Per the successor authorization, the model is swapped rather
than retried; the original L3 result remains unchanged.

Raw response recording was strengthened in the smoke harness before the next model
attempt. The contemporaneous machine-readable failure artifact is
`strategy_successor_qwen397_interface_smoke/20260718/SMOKE_FAILURE.json`.
