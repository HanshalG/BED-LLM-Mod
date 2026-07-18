# COPEx Direct-Proposal Grid-Coverage Amendment

Recorded on 2026-07-18 before any future grid-bearing comparison is interpreted or
launched.

The original `_grid_actions` implementation constructed a 12-angle grid but selected
the first three legal entries. At interior states this was a 0/30/60-degree wedge,
not the intended three directions distributed around the circle. This did not affect
the primary LLM d2 versus shared-d1 or LLM-width gates, but it invalidates the grid
d1/d2 and LLM-versus-grid descriptive comparisons in the completed direct-proposal
pilots.

The implementation now selects evenly spaced grid indices before filling any missing
legal endpoints caused by boundary clipping. For the 12-direction, three-candidate
case this yields directions near 0, 150, and 330 degrees at an interior state.
The change is covered by a regression test. Any future grid arm will use this corrected
coverage; existing LLM-only negative conclusions remain intact and are not revised.
