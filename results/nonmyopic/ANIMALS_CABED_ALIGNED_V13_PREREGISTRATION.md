# Animals CA-BED Aligned V13 Preregistration

Date frozen: 2026-07-24

V13 is the final Animals serving interface.

It preserves V12's coherent target-blind semantic response tables, exact
Bayesian planner, 64-animal support, seed 24310, smoke/formal targets,
likelihood smoothing, controls, endpoints, and formal gates.

The only protocol change is fixed candidate overgeneration before the existing
validity filter:

- root generation requests 6 candidates, then retains the first 4 valid;
- each branch requests 5 candidates, then retains the first 3 valid.

The filters are unchanged: remove history duplicates, duplicate candidates,
empty strings, and direct animal guesses. Generated menus are recorded for
audit. No regeneration is permitted when a menu is still short.

Smoke uses Wombat and Aardvark under a $0.75 hard cap and has no efficacy
threshold. Formal uses the untouched 24 targets under a $5 cap only if smoke
passes unchanged. The V10 formal conjunction remains frozen.

If serving smoke fails, the Animals line closes. There is no V14 prompt,
oversampling, model, or parser repair.
