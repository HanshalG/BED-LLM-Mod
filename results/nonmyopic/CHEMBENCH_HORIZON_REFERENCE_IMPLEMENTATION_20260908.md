# Ordinary-horizon reference: first implementation milestone

Date: 2026-09-08 (Australia/Melbourne).
Implementation commit: `6001210b`.
Scope: constructed correctness checks, not a chemistry or LLM efficacy result.
No paid model calls, cluster jobs, or chemistry response endpoints.

## Delivered

- `environments/chembench_mopen/horizon.py`: fixed-predictive-model finite-horizon
  dynamic programming, with optimized decisions at every observation-conditioned
  node. Separate exact open-loop action-sequence optimization. All supplied
  feasible actions are considered; there is no one-step screening or greedy tail.
- Fixed target weights, Bayesian conditioning on finite structure/parameter
  particles, explicit impossible-observation failures, optional repeated actions,
  deterministic index-based ties, and requested/effective horizon metadata.
- Complete selected policy trees and all first-action values, not just the chosen
  action. A zero horizon returns current risk; finite action exhaustion is explicit.
- Per-call bounded caches, cleared on success or failure; node/time/depth caps
  raise instead of falling back to a different policy. Default caps: 50,000
  expanded nodes, 30 seconds, 4,096 cache entries per cache, maximum depth 8.
  Time limits are cooperative between model operations, not process preemption.
- `scripts/chembench_horizon_reference.py`: atomic, no-overwrite report writer
  which banks execution failures separately from successful results. It records
  both planned trees and exact fixed-budget receding execution on tiny fixtures.
- `tests/test_chembench_horizon.py`: independent exhaustive policy enumeration,
  analytical Gaussian-covariance control, positive adaptivity, complementarity,
  legitimate plateaus, fixed measurement budgets, repeats, invalid inputs,
  bounded-cache equivalence, limit failures, and report overwrite/failure checks.

Existing policy-ladder code and saved scientific results are unchanged.

## Verification

Focused plus neighbouring regression tests: **89 passed in 66.46 seconds**.

```bash
uv run --with pytest==9.1.1 --with numpy==1.26.4 --with scipy==1.17.1 \
  --python /opt/homebrew/bin/python3.12 python -m pytest -q \
  tests/test_chembench_horizon.py tests/test_chembench_mopen_planner.py \
  tests/test_chembench_mopen_ir.py tests/test_chembench_mopen_source.py \
  tests/test_chembench_mopen_continuous.py tests/test_chembench_mopen_empirical.py \
  tests/test_chembench_costed_repeat_corridor.py
```

Runtime: Python 3.12.11, NumPy 1.26.4, SciPy 1.17.1, pytest 9.1.1.
The system Python lacked pytest; tests ran in an isolated uv-managed environment.
Ruff E9/F checks and Git whitespace checks passed for the new files.

Saved [RESULT.json](chembench_horizon_reference/20260908-v1/RESULT.json):
- status `reference_checks_passed`, all 10 constructed checks pass;
- SHA256 `5b5434da804d596cc701685e75fe39ecb943aff9bb1b88ac00f0cbce481d2474`;
- 68,611 bytes, with full trees and verified implementation hashes;
- maximum 49 expanded nodes among the recorded individual planning calls;
- model calls 0, model cost $0, chemistry endpoints unopened.

The XOR check selects the immediately useful noisy target at h1 (risk .1875)
and the complementary exact measurements at h2 (risk 0). h3 correctly plateaus.
The regime-assay fixture has planned h2 risk 0 for a contingent tree versus .125
for a committed pair. However, open-loop lookahead followed by REAL replanning
also reaches zero at the same two-measurement budget in that fixture. This is an
important negative control: a planned adaptation gap does not automatically
imply a deployed-policy advantage. These are analytic tests, not discovered
benchmark improvements or evidence for monotonic scientific performance.

## Source restored, without generating chemistry responses

The former temporary source checkout was absent. Restored a new sparse, durable
checkout at `external/LLM-AutoSciLab-horizon-reference` from the official
`scientific-discovery/LLM-AutoSciLab` repository. The existing read-only
`verify_source` check confirms:

- commit `acf160eb6c96897748dd92b152703b59b74efc05`;
- tree `e042d418fc30c6c70f6d1c6b0636d43f0c1c0f7a`;
- `autoscilab/oracle/chembench.py`: `eba9514d68573b4aa8c6a427606f1d0d7a9c431d8396e69ef4ae0774d2a449de`;
- `autoscilab/oracle/compound_domains.py`: `0dfd47c1858efacb732f0fbceace3ebd61bb30f864ec777d628fb70f876343f3`;
- `autoscilab/oracle/chembench_excluded.py`: `defc6c0c5edafe75dffaa366a61298856f413bd465a6e4e3636e9b0c57124003`.

The source worktree is clean. No rate-law response construction, old runtime
restoration, missing-run continuation, or old endpoint reopening was performed.
The external checkout itself is not added to this repository's commit.

## Next dependency

This completes the numerical reference milestone only. The finite categorical
provider is not a raw chemistry observation model. Next implement and test a
raw-observation particle likelihood/branch adapter with independently refined
decision values, then freeze the eight-world source-only panel's exact population,
prior, action menu, target distribution, noise, measurement budget and resource
limits BEFORE responses. The Gaussian test uses analytic covariance integration;
it is not a substitute for a nonlinear chemistry adapter.

Only a useful deployable-prior opportunity and a separate new semantic proposal
gate can justify an LLM policy pilot. No paid gate is authorized by this report.
The account-wide $5 London-day cap and no-cluster restriction remain. The research
goal is not complete and the old automation remains paused.
