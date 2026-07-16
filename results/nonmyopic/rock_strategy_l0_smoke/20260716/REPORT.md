# StrategyEIG Revival L0 Mechanics Smoke

This is a zero-LLM test of the registered Rock strategy grammar, fail-closed executor, and exact rollout-EIG scorer.

- Maps: `3-6, 5-7`.
- Trajectories: `10` (`5` per map).
- Rounds / strategies / planning horizon: `4` / `4` / `3`.
- Zero LLM calls: `True`.
- Parse / execution failures: `0` / `0`.
- All selected actions legal: `True`.
- All exact scores finite and non-negative: `True`.
- Scored roots equal executed roots: `True`.

Repository verification: `pytest -q` completed with `645 passed, 1 skipped` on
2026-07-16. The experiments-ledger validator also passed after normalizing completed
artifact paths.

**L0 mechanics gate passed: `True`.**
