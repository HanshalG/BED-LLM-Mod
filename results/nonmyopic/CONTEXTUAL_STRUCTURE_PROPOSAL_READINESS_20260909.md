# Semantic proposal readiness: public context was missing from the interface

The existing structure_proposer API accepts the seven chemistry variable names,
numeric bounds, a shared parameter prior and observations. Its system prompt
contains generic reaction-rate semantics, but no task-specific public description
can be passed. This is not an explanation of prior Luna failures: those used
different interfaces. It is a limitation of the proposed next semantic test.

Added a separate contextual_structure_proposer wrapper. PublicScientificContext
contains bounded domain, measurement and known-conditions text, with an immutable
canonical hash. The wrapper places it in the JSON data payload with an explicit
instruction/data distinction. It preserves the existing response grammar, prior,
noise semantics and original context-free prompts. History-aware and history-blind
arms receive exactly the same public context. The blind prompt remains invariant
to valid history outcomes. No transport or spending permission is added.

32 tests pass, one optional gplearn test skipped, in .70 seconds. New tests cover
context identity, blind invariance, original payload preservation, immutable
copies/hash, byte/entry limits and rejection of a hidden-equation argument.
These interface tests cannot prove that a text string is actually public or free
of answers. A frozen source manifest and explicit content audit are still needed.

## Next study design constraints

Use Luna medium, as requested, for a fresh structure-only semantic qualification;
do not repeat the closed numerical constant-correction interface. Numerical code
fits parameters. Before calls, select and pin a source with an authentic public
experimental description and executable hidden mechanisms. Keep the source's
scientifically required parameter uncertainty: do not select equations merely
because the current integrator supports two dimensions, fix unknown coefficients
to truth, or silently filter larger proposed models.

The new study should separately measure context and observation value with a
paired 2x2 proposal comparison: audited public context present/withheld, and
new-observation history/old-history control. Each fitted union receives the same
complete new history, common initial hypotheses, parameter prior, proposal-call
budget and sealed prediction inputs. Numeric regression and a productive symbolic
competitor must remain in the comparison; an LLM win over a blind LLM alone is
insufficient. Do not use a condition label or source-equation ID as public context.

Prospectively bind cohort, public text, priors, endpoints, thresholds and full
worst-case costs only after source and inference compatibility have been checked.
The parser must either retain every valid proposed structure or fail the complete
case; unresolved inference is not permission to select survivors. Capability
qualification comes before an efficacy or depth run. Any later planner requires
a joint predictive distribution and the same realized/imagined update, not the
current marginal moment object.

This document is a readiness decision, not an executable paid protocol. Source,
cohort and a compatible full parameter domain are still missing. No paid request
is authorized by this wrapper, and no old chemistry/Bongard/Number Game gate is
reopened. Avoid another broad benchmark hunt: next inspect one source contract
against the public-context/hidden-mechanism distinction before building a runner.

Previous turn was progress. This turn adds a tested missing interface boundary;
it does not establish useful LLM discovery or a non-myopic positive. Goal remains
active. No new calls or cost; authenticated daily remaining $4.11174654 and balance
$23.693468061 unchanged. No cluster or automation changes.
