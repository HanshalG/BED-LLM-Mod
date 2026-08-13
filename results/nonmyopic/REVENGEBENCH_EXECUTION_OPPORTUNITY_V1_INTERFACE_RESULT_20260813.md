# RevengeBench execution-opportunity V1 interface result

Date: 2026-08-13

Status: **infrastructure inconclusive before trajectories**

OpenRouter calls/cost: **0 / $0**

## Finding

The frozen V1 protocol requires every candidate target strategy to act on every
other target's frozen states. This is natively executable for BattleSnake,
Halite, and HuskyBench using the release's own policy interfaces and action
distance parsers.

It is not executable as written for RoboCode. Frozen RoboCode targets are
stateful Java engine bots (`MyTank.java`). The release's offline RoboCode
evaluator does not counterfactually invoke these target bots on an arbitrary
state. It invokes a separate learned Python `main.py:move(state)` surrogate,
which is not present in the frozen target bank. Treating an observed Java action
or a source heuristic as the missing counterfactual would change the likelihood
model after freeze.

No selected source content, opportunity trajectory, endpoint state, provenance,
or prior outcome was opened. V1 is therefore banked as an interface-level
infrastructure incompatibility, not a scientific horizon null.

## Successor rule

A prospective V2 may retain only the three arenas with genuine native
counterfactual policy interfaces. It must preserve the original scientific bar:
at least two arenas must have a changed depth-two first probe at every beta, each
with at least `0.01` nat gain and held-out endpoint separation. No threshold,
target, probe, seed, beta, parser, or distance may be tuned after trajectories.
