# Range-Gated Rock Depth-Four Qwen 14B Preregistration

Status: frozen before any Qwen depth-four response.

## Rationale and Separation

The exact corner-start h4 qualification passed with a `+0.348436` entropy-AUC
gain over exact d3. The first fixed-tail transfer, Gemma 4 26B A4B thinking,
returned legal plans but missed the three-move on-site route in `10/10` cells.
Its common error stopped one cell short of a rock.

This is a separately reported model replication using dense Qwen 3 14B
(`14.8B` parameters) with thinking. It cannot replace, pool with, or relabel the
failed Gemma line. Qwen previously received the h3 successor-grounded prompt
under the old reasoning-only OpenRouter adapter and produced no accepted cell.
The adapter now has an endpoint-free calibrated forced-final path, but Qwen has
not received this h4 prompt, state, route, or endpoint.

## Frozen Interface

The h4 environment, K4 roots, prompt, parser, dynamic legality checks, exact
scoring, controls, and thresholds are byte-for-byte the committed
Gemma protocol. Only model identity and fresh seeds change.

- Qwen 3 14B thinking.
- 4,096-token reasoning allowance and one bounded 256-token
  reasoning-disabled finalization.
- Temperature zero and one registered correction attempt.
- Score-free reachable-position graph, rock coordinates, sensor law, and
  posterior marginals.
- Three explicit tail actions for every machine-fixed root.
- Exact open-loop EIG scoring with no model calls.
- `$1` hard run cap per stage; projected combined cost below `$0.50`.

## S0: Serving and Route Smoke

- Fresh seed `24212`.
- Ten distinct belief cells.

S0 passes only if:

1. all ten cells return four dynamically legal fixed-root plans;
2. every accepted, invalid, forced-exit, forced-final, and correction request is
   retained and accounted for;
3. `move-NORTH, move-NORTH, move-NORTH, check-4` appears in at least `8/10`
   cells;
4. exact scoring selects the exhaustive-h4 root in at least `8/10` cells; and
5. exact scoring makes no model calls.

Any failure stops this Qwen line without repair or replacement.

## Conditional S1: Proposal Quality

S1 runs only after S0 passes.

- Fresh seed `24213`.
- Sixteen distinct strict h4-over-h3 cells.
- Producer bootstrap seed `24214`; 5,000 paired replicates.
- Independent audit bootstrap seed `24215`.

The frozen controls are identical-root random h4 tails, shared-plan h3 scoring,
the strongest exact d3 root with its best h4 continuation, and exhaustive h4.
The producer and independent audit must both satisfy:

- positive 95% lower bounds versus random h4, shared h3, and strong d3;
- at least 75% complete exhaustive-h4 route selection;
- at least 60% mean exact opportunity recovery; and
- complete legal/mechanical/usage replay with no scoring-time model calls.

A passed S1 would establish model-replicated h4 proposal quality only. It would
authorize, but not itself constitute, a separately preregistered trajectory
confirmation.
