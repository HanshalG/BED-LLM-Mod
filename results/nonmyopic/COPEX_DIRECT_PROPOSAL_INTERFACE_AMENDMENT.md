# COPEx Direct-Proposal Interface Amendment

Recorded on 2026-07-18 after the first interface-only serving probe and before any
policy endpoint was read.

## Trigger

Run `copex-direct-proposals-serving-probe-20260718b` made two non-thinking Gemma 26B
requests (`$0.00016375`, zero reasoning tokens) and failed closed at its first root
cell. Its first JSON was malformed; the one permitted correction was valid JSON but
used raw `dx/dy` values above the registered `0.1` L-infinity bound. No trajectory,
selection, policy endpoint, or comparison from this probe was inspected or used.
The raw failure record remains at
`results/nonmyopic/copex_direct_proposals_serving_probe/20260718b/FACTORIAL_FAILURE.json`.

## Change

The LLM interface changes from raw numeric movement offsets to a JSON cell:

```json
{"angles_deg":[0.0,133.5,271.0]}
```

Each angle is finite, distinct, and in `[0,360)`. The program converts it to the
maximum legal L-infinity move (`0.1`) in that direction, clipping only at the task's
`[0,1]^2` boundary and rejecting no-op or duplicate endpoints. This preserves the
same continuous action space, LLM-only proposal role, posterior/scorer, d1/d2 tree,
grid control, and width-call allocation from the preregistration. It removes only
the fragile raw magnitude arithmetic that caused the interface failure.

The prior COPEx width interface used the same bounded-angle encoding successfully;
this is a mechanics repair, not an outcome-driven change. The eight-trial pilot is
reregistered with all numerical parameters, seed, model, endpoints, and promotion
rule unchanged.
