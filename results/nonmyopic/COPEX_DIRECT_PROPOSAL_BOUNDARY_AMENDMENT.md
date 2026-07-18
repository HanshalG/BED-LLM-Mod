# COPEx Direct-Proposal Boundary Amendment

Recorded on 2026-07-18 after the preregistered pilot failed closed and before a
replacement pilot is launched or any policy endpoint is read.

## Trigger

The first pilot (`copex-direct-proposals-pilot-20260718`) made 1,147 requests for
`$0.05198270`, with zero reasoning tokens. It failed closed after a late cell returned
three distinct direction angles whose box-clipped endpoints were not distinct. No
completed trajectory, endpoint, or comparison was produced. Its full failure artifact
is retained at `results/nonmyopic/copex_direct_proposals_pilot/20260718/FACTORIAL_FAILURE.json`.

## Change

Three distinct LLM angles remain mandatory. The executor now maps every angle to its
legal clipped endpoint and removes only endpoints that are no-ops or duplicates. It
does not insert, repair, or sample replacement actions. A cell with no realizable
endpoint still fails closed.

The d2 tree uses the resulting `K_realized` actions and therefore allocates
`1 + K_realized * outer_rollouts` proposal calls at a nonterminal root. The width
control receives exactly that same state-local logical allocation. This corrects an
artifact of the angle parameterization at the physical boundary; it does not change
the continuous task, particle posterior, score, model, seed, endpoints, or grid arms.

The replacement pilot uses the same frozen seed and all other numerical settings.
Its result is the first outcome-bearing result for this amended interface.
