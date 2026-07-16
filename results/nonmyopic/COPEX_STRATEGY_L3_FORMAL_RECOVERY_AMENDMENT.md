# COPEx-Task L3 Formal Recovery Amendment

Recorded on 2026-07-16 after the seed-31002 formal invocation failed closed and before
one fresh-seed replacement. The failed invocation produced no `L3.json`, arm endpoint,
or gate result. Only failure classes and usage were inspected.

The invocation made 1,328 requests (507,929 prompt tokens, 524,020 completion tokens,
zero reasoning tokens, one forced exit, `$0.24509260`) before a width cell repeated a
vector after its single feedback retry. It had 1,237 accepted cells and 91 rejected
attempts. Eighty-eight of the rejects were width serialization failures, dominated by
malformed long vector JSON; only three were strategy cells.

The final interface repair changes only the width-cell wire format. Instead of 16
`{"dx":...,"dy":...}` objects, the LLM returns 16 distinct direction angles in
`[0,360)`. The constrained compiler maps each selected angle to its unique maximum
legal L-infinity step. This is the same deterministic action-macro compilation used by
the plan grammar: no angle is inserted, dropped, deduplicated, padded, or substituted.

The width arm still uses one LLM call, proposes the same number of continuous roots,
and receives exactly StrategyEIG's rollout scorer units. All scientific settings,
controls, likelihoods, endpoints, and the intersection gate remain unchanged. The
replacement uses fresh seed `31003`. This is the final pre-endpoint recovery; another
interface failure closes L3 and triggers consolidation without a further run.
