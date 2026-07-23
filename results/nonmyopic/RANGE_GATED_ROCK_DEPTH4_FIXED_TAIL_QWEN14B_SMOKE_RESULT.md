# Range-Gated Rock Depth-Four Qwen 14B Smoke Result

Status: **failed closed on cell 0; conditional S1 was not run.**

Dense Qwen 3 14B thinking did not complete the frozen all-branches-legal
interface. The first response exhausted the reasoning/output allowance; its
bounded finalization also exhausted 256 tokens and returned no usable JSON. The
registered correction call then returned a complete object.

That corrected object contained the exact critical north-root plan:

```text
move-NORTH, move-NORTH, move-NORTH, check-4
```

However, two other root tails were dynamically illegal. For example, after the
fixed `check-0` root it proposed `move-WEST, move-SOUTH, check-5`; the south move
from `(5,6)` leaves the grid. Because the frozen compiler requires all four
branches to be legal, cell 0 had zero accepted plans and no exact score or policy
endpoint.

## Usage

- accepted cells: `0`
- invalid responses: `2`
- physical requests: `3`
- forced exits: `2`
- forced-final requests/successes: `1/0`
- prompt tokens: `20,813`
- completion tokens: `8,834`
- reasoning tokens: `8,163`
- cost: `$0.00805743`
- project spend after S0: `$38.56821252 / $110`

Seed-24213 S1 is permanently unauthorized. Relative to the Gemma S0, this
failure is diagnostically different: the dense model composed the load-bearing
four-action route but could not serialize every comparison branch legally. That
supports a separately preregistered bounded-projection mechanism test in which
valid branches are preserved exactly and only invalid branches receive a
deliberately non-routing legal tail. It does not permit retroactive scoring of
this failed response.

Artifacts:

- `range_gated_rock_depth4_fixed_tail_qwen14b_smoke_20260723/SMOKE_FAILURE.json`
- `range_gated_rock_depth4_fixed_tail_qwen14b_smoke_20260723/run.log`
