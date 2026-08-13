# Atomic-Particle Half-Bank Clarification

Date frozen: 2026-08-13, before implementation completion and before any model
response under the atomic-particle interface.

The parent protocol fixes exactly one adaptive second query for each
tree/root/first-answer node before second-refresh requests are constructed.
Consequently its half-bank stability gate is interpreted as estimator
stability on that same target-blind full-bank query topology:

- split every initial, first-refresh, and second-refresh request group by even
  versus odd request slot;
- halve all particle widths from 64 to 32, preserving the same 50/50 retained
  versus generated composition;
- keep the four full-bank candidate roots and each full-bank dynamic second
  query fixed; and
- independently recompute every answer probability, entropy, continuation
  value, root score, and selected root from each disjoint half.

The half estimators may not choose a different second query because no
response under that counterfactual query was requested. This is a stability
audit of the Monte Carlo root-value estimator, not an independently executed
half policy. Any reachable half branch with no valid retained or generated
particle receives a failed stability value; it is never repaired or borrowed
from the other half. All other gates, calls, seeds, thresholds, costs, and
authority remain unchanged.
