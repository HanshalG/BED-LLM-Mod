# Prospective finite-prior physical-command opportunity screen

Scope: all8groups from MQTT_SOURCE_METADATA_20260909.json, including singleton,
bound to hash b2ddd4eb02218334f6d00331e87b0b8ca9652c483f4452873e4b4b2479598e62.
Uniform prior across published machines within each group; equal weight across
groups in descriptive aggregates. Do not remove duplicates or low-opportunity
groups. This prior is a diagnostic assumption, not a calibrated deployment law.

Each episode starts in each candidate machine's designated initial state. One
action is one command from its common alphabet, or explicit RESET. Each costs
one unit, including RESET; reset yields a fixed public acknowledgment and resets
the physical state, not the accumulated posterior. Initial setup is common and
free. These are declared simulator costs, not empirical hardware latencies.
Budget B=6. Repeated commands allowed. No free membership words, no equivalence
oracle, no unknown-command completion, no direct state observations.

Targets:512 reset-start command words of length6 per group, sampled independently
with replacement from its sorted input alphabet using random.Random(61400+group
index), group order exactly metadata order. Bank these input-only menus before
any target-response calculation. Target outcome is the complete six-output trace;
primary terminal risk is mean normalized whole-trace Brier Bayes risk on this fixed
menu under the declared source prior. Targets never depend on chosen experiments
or realized responses. Do not substitute machine-ID accuracy or reward for risk.

Use exact deterministic likelihood and rational arithmetic where feasible. Compare
ordinary receding h1/h2/h3 at the same6-command budget, adaptive optimum for B6,
an optimal committed open-loop6-command sequence (including RESET), and random
commands. The committed control may inspect its outputs and update its posterior,
but cannot change subsequent commands. This is stronger than a deliberately weak
one-symbol-myopic baseline and guards against mere access-prefix benefits.
Also report initial risk, first actions, reachable posterior sizes and work/time.

Freeze a180second per-group reference cap before execution; incomplete groups
stay incomplete, never excluded from an aggregate pass or retried with a larger
cap. Verify planning mechanics independently on tiny synthetic cases before any
benchmark scoring. Replay source hashes and menu coverage. No runtime change or
threshold rescue after results. Mark singleton risk0 as trivial, not success.

Opportunity passes only if all8groups complete and all controls verify; mean h2
improves on h1 by >=5%, h3 on h2 by >=5%, and adaptive B6 improves on the optimal
committed sequence by >=5%, with at least3nontrivial groups exhibiting positive
adaptive-versus-committed gain. Relative denominators must be positive. Report
all individual plateaus/reversals even if aggregate criteria pass. This criterion
addresses prospective practical headroom, not statistical population significance.

A pass authorizes only designing a new LLM semantic/joint-predictive and actual
versus simulated regeneration gate. No supplied-model classification experiment
counts as an LLM-native result. A null closes this precise finite-prior/control/
budget panel; do not rotate targets, remove models, rename horizon, or tune cost.
No LLM calls or new inference budget is authorized by this protocol.
