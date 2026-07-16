# Strategy-Prior Protocol Audit

Audited: 2026-07-17. This is a terminal evidence audit only: it made no model calls,
policy reruns, or changes to the registered endpoint.

## Determination

**The objective is not established.** The objective required evidence that the LLM's
natural-language strategy prior makes non-myopic scoring load-bearing, rather than
merely supplying useful candidate plans. The continuous L3 intersection gate is false:
the paired 95% lower endpoints for StrategyEIG against both grammar-matched random
strategies and shared-d1 are not strictly positive. Under the recorded protocol this is
an honest negative/partial result, not a justification for another rescue run.

## Requirement Check

| Requirement | Evidence | Status |
| --- | --- | --- |
| L0 grammar, fail-closed executor, exact scorer, zero-LLM mechanics | `rock_strategy_l0_smoke/20260716/REPORT.json`: complete candidate sets, legal actions, finite non-negative exact scores, zero parse/execution failures, and zero LLM calls | Satisfied |
| Rock L1 anchor with all five arms and exhaustive-d2 fraction | `rock_strategy_l1_confirmation/20260716/L1_FAILURE.json` and its sole registered retry: fail-closed before an arm endpoint | Unavailable, not a failed policy comparison |
| L3 continuous non-enumerable plan-space endpoint | `copex_strategy_l3_confirmation_recovery1/20260716/L3.json`: 30 paired trials, 30 rounds, K=4, horizon 4, 64 particles plus truth, and 64 CRN rollouts | Completed |
| Required L3 controls and mechanics | Same L3 artifact: shared-d1, equal-compute width, grammar-matched random strategies, matched-budget grid-d2; legal actions, shared initial strategy/d1 cells, matched scorer units, and zero rollout-scoring LLM calls | Satisfied |
| LLM-primacy / non-myopic gate | L3 paired final-entropy CI: versus random `[-2.404e-63, +0.004548]`; versus shared-d1 `[-1.353e-55, +0.0000287]`; `gate_passed: false` | Failed |
| Quantify the enumerable classical crutch | `copex_strategy_l3_grid_sensitivity/20260716/GRID_SENSITIVITY.{json,md}` varies the full d2 grid tree from 16 to 256 sequences while keeping eight evaluated sequences | Satisfied for L3; no Rock fraction exists because L1 has no endpoint |
| Preserve verbatim strategies and reproducibility data | The L3 JSON's `requests` array contains all 1,371 accepted raw model responses, plus paired outcomes, configuration, and token/cost usage | Satisfied |
| Test whether a simulator-ranking mismatch explains L3 | `copex_strategy_l3_ranking_fidelity/20260716/REPORT.json`: 900 recorded cells / 3,600 plans, zero LLM calls or reruns; trial-level score/realized fixed-plan entropy Spearman `0.653551` | Completed posthoc diagnosis; does not change the gate |
| Repository and ledger validity | `pytest -q`: `665 passed, 1 skipped` on 2026-07-17; `python scripts/validate_experiments_ledger.py --json`: all checks pass, no active rows, complete artifacts exist | Satisfied |

## Consequence

The evidence supports a narrower statement: the LLM supplies a useful continuous-plan
proposal prior, and the exact rollout scorer has meaningful fixed-plan rank fidelity.
It does not support the stronger statement that horizon-4 receding-horizon strategy
scoring improves the deployed root action beyond shared myopic scoring or matched
random legal plans. The companion
`STRATEGY_PRIOR_RESULT.md` is the paper-facing synthesis; this audit makes the
requirement-to-evidence chain explicit.

`STATE.md` records the terminal decision: Rock consumed its allowed interface retry,
the sole 31B reasoning probe was unavailable after fail-closed JSON absence, and no
further StrategyEIG policy run is authorized under this protocol.
