# Initial Number Game beliefs: anticipation-of-adaptivity control

Date: 2026-09-08. Zero new model calls; $0 incremental inference cost.
Historical generation cost is not zero and is not reassigned to this audit.

## Scope

This is a retrospective descriptive control on the same 32 banked initial
LLM-generated supports as NUMBER_GAME_INITIAL_HORIZON_RESULT_20260908.md.
Protocol and executable were committed/pushed at `7ef8633f` before evaluation.
It opens neither historical future proposals nor hidden target rules.
All supports, uniform weights, deterministic observations, three-query budget,
and fixed 101-number Brier target set remain unchanged.

The parent `initial_opportunity_null` remains unchanged. This analysis does not
create a substitute gate, authorize a paid call, or rescue its failed >=5%
successive-depth condition.

## Controls

- **Committed three:** optimize a fixed set of three queries before any answers.
- **Receding open-loop h3:** optimize a fixed set for the remaining budget,
  execute the first query, observe its real answer, then replan. It uses feedback
  but does not anticipate answer-dependent future actions in each planning pass.
- **Contingent h3:** optimize the full answer-conditioned three-query tree,
  using the exact saved value from the parent audit.

Fixed deterministic query order does not affect committed risk. Sets are sorted
and ties are resolved lexicographically after deduplicating equivalent/complement
columns and omitting constant columns. The receding control executes the smallest
query in its selected set. This is a specified deterministic open-loop control,
not an optimal choice of tie-breaking under its eventual receding objective.
Queries remain in the target score. Filler measurements after a resolved posterior
have no effect on risk; redundant measurements cannot improve deterministic inference.

## Complete results

| Policy | Mean expected terminal Brier |
|---|---:|
| Committed three | 0.0670482261 |
| Receding open-loop h3 | 0.0651116705 |
| Contingent h3 | 0.0631955085 |

Contingent h3 versus receding open-loop h3: **2.9429%** lower aggregate loss,
**25 wins / 7 ties / 0 losses**. Contingent h3 versus committed three:
**5.7462%** lower loss, **30 wins / 2 ties / 0 losses**.
The h3 <= receding <= committed exact-risk inequalities held on every support.

All 32 completed in 75.035 seconds. Maximum per-support time was 3.534 seconds;
maximum enumerated query sets was 150,511, within frozen 60-second/500,000-set
per-support ceilings. No runtime/cap adjustment or repeat run occurred.

## What changed scientifically

The initial-support result contains genuine value from anticipating adaptive
continuations, not merely selecting a better precommitted query set. Replanning
after observations explains some, but not all, of the advantage over commitment.
This complements the earlier finding that all 21 h3-versus-h2 gains sacrificed
two-step utility rather than simply breaking a two-step tie.

The comparison is exact conditional on the saved finite beliefs and specified
ties. It is not a fresh-world generalization estimate, proof of calibrated LLM
beliefs, proof of LLM necessity versus symbolic proposals, or a demonstration of
anticipating future LLM model discovery. The 32 initial sets come from an existing
bank, not a newly sampled population. No population significance claim is made.
The original 1.9735% h2-to-h3 gain remains below its frozen threshold.

The next scientific dependency is useful, calibrated model discovery with fair
symbolic/history-blind controls under a genuinely new prospective protocol. These
descriptive controls alone do not authorize that paid stage. The overall research
plan is unfinished and the automation remains paused.

## Verification and artifacts

18 focused tests passed in 1.31 seconds across initial-horizon, mechanism, and
open-loop tests. Independent explicit outcome grouping verifies fixed-sequence
values; independent real-history recursion verifies the deployed receding control.
A constructed example separates committed and contingent two-query risk.
The source solver/compiler and initial/result hashes were checked before output;
every saved rule's extension hash and positive count were replayed.

Result: `number_game_initial_openloop_audit/20260908-v1/RESULT.json`

- Result SHA256: `ffab610fa902c82d2a14fb7ec02532b519a063c2c54daeb4e57a40ac94a79ae3`.
- Protocol SHA256: `1b9d448824f7355557ee64138830dce68b352dc3d4ffbd5a8b5eb56a80936429`.
- Runner SHA256: `bfe2e8fc7ec6efa219a2c9b5d5443289ddbdf53ca4131422de5f9e180d342af4`.

Per-support exact rational values, chosen sequences, and runtime/work counts are
banked beside the aggregate result. No frozen parent artifacts were edited.
