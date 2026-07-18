# COPEx Direct-Proposal Angle Serving Probe

Run `copex-direct-proposals-angle-serving-probe-20260718` was an interface-only
probe after the angle-cell amendment and before the preregistered pilot. It used one
trial and two queries. Policy endpoints are quarantined and are neither reported nor
used for any parameter, model, or decision change.

## Mechanics

- Completed with 26 physical OpenRouter requests and 30 logical proposal calls.
- Accepted cells: 26; rejected cells: 0; terminal cell failures: 0.
- All selected actions were legal.
- The initial d1 and d2 root cells were identical.
- Current-state width allocation matched the depth-two virtual allocation.
- Reasoning tokens: 0; forced exits: 0.
- Cost: `$0.00110346` (9,978 prompt and 734 completion tokens).

The full raw result is retained at
`results/nonmyopic/copex_direct_proposals_serving_probe/20260718c/FACTORIAL.json`
for audit, but it is not an outcome-bearing experiment.
