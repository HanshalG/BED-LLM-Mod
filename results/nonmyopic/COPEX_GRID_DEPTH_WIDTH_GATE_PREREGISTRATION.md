# COPEx Grid Depth-Width Gate Preregistration

Frozen on 2026-07-18 before this zero-LLM gate is run.

The corrected angular grid now exposes a necessary environmental check before further
LLM proposal work. Using the same continuous COPEx equations, 48-plus-truth finite
support, three-node Gaussian quadrature, eight stratified child branches, eight
queries, and shared trajectory noise, compare:

1. `grid_d1`: immediate EIG over three evenly distributed angular directions.
2. `grid_d2`: depth-two EIG over three root and three child angular directions.
3. `grid_score_width`: immediate EIG over `K^2 * B = 72` evenly distributed angular
   directions. At a box boundary, where clipping would make some full-step angular
   directions identical, the control fills the legal action set with shorter radial
   steps on the same angular lattice. This matches d2's root-by-branch-by-child count
   of quadrature candidate-action evaluations at each nonterminal decision.

Run 100 paired zero-LLM trials at fresh seed 46023. Primary endpoint is entropy-AUC
reduction `H_width - H_grid_d2`; positive values favor depth two. A clearly nonpositive
mean rejects this COPEx action geometry as a compute-matched-width setting, regardless
of prior d2-versus-narrow-d1 observations. A positive result authorizes an LLM-root,
programmatic-child hybrid screen; this gate itself makes no LLM claim.
