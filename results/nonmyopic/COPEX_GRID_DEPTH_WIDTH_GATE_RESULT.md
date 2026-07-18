# COPEx Grid Depth-Width Gate Result

Run on 2026-07-18 from the preregistered protocol in
`COPEX_GRID_DEPTH_WIDTH_GATE_PREREGISTRATION.md`.

## Protocol

- 100 paired trials, seed 46023, eight decisions per trial.
- 48-plus-truth finite support, exact three-node Gaussian quadrature, and eight
  stratified outer branches.
- Shared truth trajectories and observation noise across arms.
- The angular grid was corrected to cover the circle. The score-width control
  considered 72 legal immediate actions per nonterminal decision: the same
  root-by-branch-by-child candidate-score count as depth two. At boundaries it
  uses shorter radial steps only when full-step directions clip together.
- `--dry-run`: the direct-proposal model was deterministic; there were no network
  requests and zero cost. The logged proposal-cell counts are simulated accounting
  cells, not paid LLM requests.

## Result

The primary contrast, depth two minus compute-matched one-step width, was
`-0.3676` nats entropy-AUC, with a paired 95% bootstrap interval
`[-0.5005, -0.2378]` and 24 / 0 / 76 wins / ties / losses. The wide one-step
control also had lower final entropy (`0.6709` versus `1.0209`) and lower final
RMSE (`0.0475` versus `0.0951`).

For reference, narrow grid depth two minus narrow grid depth one was `-0.0648`
nats, 95% CI `[-0.1429, +0.0138]`. All actions were legal, the initial root
cell sharing check passed, and the original LLM-width accounting invariant passed.

Raw artifacts: `copex_grid_depth_width_gate/20260718/FACTORIAL.json` and
`copex_grid_depth_width_gate/20260718/FACTORIAL.md`.

## Decision

Reject the current continuous COPEx geometry as a compute-matched-width
demonstration of non-myopic value. In particular, do not run the planned
LLM-root/programmatic-child hybrid in this geometry: a stronger one-step action
set has already beaten depth two without any LLM. The earlier apparent
grid-depth advantage came from a defective narrow angular grid and is superseded
by this result.
