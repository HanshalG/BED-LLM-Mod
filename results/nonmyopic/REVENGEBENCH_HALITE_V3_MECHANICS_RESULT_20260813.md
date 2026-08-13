# RevengeBench Halite V3 mechanics result

Date: 2026-08-13

Status: **infrastructure inconclusive before planner scoring**

OpenRouter calls/cost: **0 / $0**

## Finding

The first paired Halite engine cell is byte-exact across fresh arms. However,
the release's public offline feeder disagrees with the true compiled target on
26.46% of per-cell actions, so V3 fails the mandatory diagonal self-distance
gate before any likelihood or planner score.

The public C SDK explains the mismatch. `GetInit()` consumes an initialization
map. Every bot then begins its action loop by calling `GetFrame()`. To reproduce
recorded move 0, an offline feeder must therefore send replay frame 0 again after
reading the bot name. The release feeder instead sends frame 1 and compares the
response against move 0, shifting every policy decision by one frame.

A mechanics-only diagnostic on the failed V3 replay confirms the correction:
the exact true bot changes from 26.46% mean distance to zero, while the other two
hypotheses retain distances 42.68% and 17.42%. This diagnostic is not a V3
scientific result and cannot authorize use of the old trajectory.

## Successor

V4 may change only the Halite offline feeder to send frame `i` before evaluating
recorded move `i`, including resending frame 0 after initialization. All V3
hypotheses, probes, engine seeds, replay mechanics, parsers, distances,
likelihoods, planners, and thresholds remain unchanged. V3 Halite trajectories
are discarded.
