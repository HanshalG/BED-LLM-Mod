# Isolated symbolic control and weighting readiness

The pinned container executed the generic23-operation first-order beam on a
hand-built mirror demonstration, with no RE-ARC source task or held-out examples.
It attempted25,743calls, recorded5,317invalid candidates, and returned the correct
single-step vmirror graph with training loss0. Other retained behavior classes
had loss.625. Runtime used UID65534 and no API key, a read-only mount of required
modules only, no network,512MiB memory ceiling,20CPU-second and60wall-second limits.
This is a comparator-specific larger compute cap than individual program execution.
Its first-order/6step search limitations remain explicit, not a fullDSL baseline.

Added condition_programs: uniform prior over unique canonical graph JSON,
deterministic full-demonstration likelihood, and normalized mass on consistent
programs only. Duplicates cannot inflate prior mass; inconsistent duplicate replay
raises. Runtime failures have zero likelihood. If every program fails consistency,
the result explicitly requests a unit failure forecast, which incurs maximal
predictive loss rather than resetting the prior or omitting the task.

This is exact conditioning WITHIN a declared finite pool, not correction for
data-dependent LLM proposal selection. Predictive qualification remains necessary.
The same rule must apply to initial-only, initial-plus-aware, initial-plus-blind,
and symbolic pools, using the SAME complete demonstration history at evaluation.

Next frozen example split: for each retained task, seeds31100..31102 are the three
demonstrations; seeds31200..31207 are eight held-out predictions. Initial and blind
prompts receive only demonstration0's answer; aware refresh receives all3answers.
All arms see the same3demonstration input grids. Held-out inputs/outputs are absent
from every proposal prompt. One attempt per seed; any source mismatch fails the
panel, not a substituted draw. These seeds are fixed here before generation.

The proposed call structure is one4-program initial call per task, then paired
4-program aware and blind calls, twelve calls total if preceding serving checks
pass. All pools are scored after conditioning on the full3demonstrations. Before
dispatch the complete paid protocol still needs literal thresholds, request caps,
ledger reservations, bank/replay behavior and live provider schema qualification.
This document alone authorizes no model call or held-out outcome opening.

Previous turn was symbolic mechanics progress; current turn exercises it in
isolation and fixes the common weighting and future example split. Cost$0,
balance23.693468061, daily remaining4.11174654 unchanged. Goal active/unachieved.
