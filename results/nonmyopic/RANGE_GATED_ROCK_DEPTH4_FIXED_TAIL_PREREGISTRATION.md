# Range-Gated Rock Depth-Four Fixed-Tail Preregistration

Status: frozen after deterministic instrument and audit tests, before any live
depth-four model response.

## Motivation and Development Boundary

The seed-24201 exact qualification established a strict fourth-step effect in the
standard RockSample[7,8] layout with only the rover start changed to `(6,6)`.
Exact d3 checks remotely, while exact d4 takes
`move-NORTH, move-NORTH, move-NORTH, check-4` and gains `+0.348436` entropy AUC
over d3 across 500/0/0 paired wins/ties/losses.

Zero-LLM development screens used seeds `24204` and `24205` to verify that the
candidate belief-cell generator retains this strict opportunity. Those cells and
seeds are excluded from both live stages below. No live h4 model response has been
observed.

## Frozen Interface

- Model: OpenRouter Gemma 4 26B A4B thinking.
- Reasoning budget: 4,096 tokens with one bounded 256-token reasoning-disabled
  finalization when the provider returns reasoning-only at the length cap.
- Temperature: zero.
- One registered correction attempt after an invalid response.
- Exact JSON-prefix parser and dynamic four-action legality checks.
- Machine-fixed K4 roots: the two geometry-nearest legal moves and two
  highest-immediate-EIG checks.
- The model supplies exactly three tail actions for every root.
- The prompt provides rock coordinates, current posterior marginals, sensor
  accuracies, and a reachable-position transition graph. It supplies no utility
  scores, route labels, or preferred action.
- Exact open-loop EIG scores all accepted plans. Scoring makes no model calls.
- Separate hard OpenRouter cap: `$0.25` per stage.

## S0: Serving and Route Smoke

- Fresh seed `24208`.
- Ten distinct fixed belief cells.
- At most ten accepted logical cells and one correction attempt per invalid cell.

S0 passes only if:

1. all ten cells return four legal plans with the exact machine-fixed roots;
2. all accepted, invalid, forced-exit, and forced-final requests are accounted for;
3. the exact route `move-NORTH, move-NORTH, move-NORTH, check-4` is present in at
   least eight cells;
4. exact scoring selects the exact-h4 root in at least eight cells; and
5. scoring makes no model calls.

Any failed S0 condition stops the h4 proposal line without a prompt, parser,
threshold, budget, or seed repair.

## Conditional S1: Proposal-Quality Gate

S1 runs only after S0 passes.

- Fresh seed `24209`.
- Sixteen distinct strict h4-over-h3 opportunity cells.
- Producer bootstrap seed `24210`, with 5,000 paired replicates.

Every LLM proposal set is compared to:

1. **Identical-root random h4:** the same four machine-fixed roots with three
   uniformly sampled legal tail actions.
2. **Shared-plan h3:** truncate each LLM h4 plan after action three, select with
   exact h3 EIG, then score the corresponding full h4 plan. This isolates the
   extra planning step while holding the LLM proposal set fixed.
3. **Strongest exact d3 root:** select the exact exhaustive d3 root, then give
   that root its best exhaustive h4 continuation.
4. **Exact exhaustive h4:** the full open-loop oracle used to measure opportunity
   recovery and route selection.

The producer passes only if:

- the paired 95% bootstrap lower bounds for LLM-minus-random,
  LLM-minus-shared-h3, and LLM-minus-strong-d3 are all strictly positive;
- exact scoring selects the complete exhaustive h4 route in at least 75% of cells;
- mean recovery of the exact-h4-over-strong-d3 opportunity is at least 60%;
- all 16 cells are distinct strict opportunities with the registered critical
  route;
- all roots, plans, controls, and usage are fully accounted for; and
- exact scoring makes no model calls.

## Independent Audit

The independent audit uses fresh bootstrap seed `24211`. It must reconstruct each
posterior from serialized history, validate every plan, regenerate every random
control, recompute the shared-h3 and strong-d3 controls, re-enumerate exact h4,
match every record and producer aggregate, and obtain fresh intervals satisfying
the same endpoint gates.

This protocol can establish that an LLM proposal interface supplies a
load-bearing fourth planning step. It does not authorize a trajectory claim; any
receding-horizon h4 policy evaluation requires a later preregistration.
