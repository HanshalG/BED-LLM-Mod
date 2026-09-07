# Executable model proposals: zero-call inference connection

## Why this change

The previous goal turn produced actionable negative evidence: the completed
calibration-context audit failed the prespecified planning-headroom thresholds.
That exact formulation remains closed. This change does not rerun it, expand
its assay menu, alter its prior, or authorize an LLM experiment.

The remaining plan requires actual executable LLM-generated model spaces.
Inspection found an existing allowlisted rate-law representation in `ir.py`,
but no connection from arbitrary proposed expressions to the new raw-observation
receding-horizon planner. The old registry-edit atlas is a different, closed
interface and is not reused as an LLM gate.

## Implemented instrument

`environments/chembench_mopen/executable_belief.py` now provides:

- A bounded pool of proposed expressions using the existing rate-law syntax,
  without requiring membership in a hidden or public mechanism registry.
- Batched floating-point evaluation of the validated arithmetic tree, without
  executing generated Python. Syntax, expression size, parameter count,
  intermediate numerical values, negative rates, work and workspace are checked.
- Deterministic parameter draws from each law's declared log/identity uniform
  prior. The snapshot retains parameter uncertainty, not a fitted point estimate.
- Equal prior mass across unique canonical laws. Names, rationales, duplicate
  submissions and submission order do not change the prior mass or draws.
  Deduplication is the existing syntactic canonicalization, not a proof of
  algebraic equivalence for arbitrary formulas.
- Full Gaussian likelihood on log1p rates, replaying the entire supplied real
  history exactly once from the prior after refresh. New laws start with prior
  mass before conditioning; old observations are not counted twice.
- A history-bound snapshot with log weights, fixed target predictions, parameter
  draws and work accounting. A later pool refresh leaves earlier snapshots
  unchanged. Genuine h1/h2/h3 planners can consume these snapshots with no
  hypothesis generation inside imagined branches.

The adapter accesses no source code registry, hidden-world constructor, network
or LLM. Invalid inference-domain behavior fails the snapshot; it does not quietly
drop particles or condition on only the completions that happen to work.

## Verification

Focused tests cover independent manual Gaussian posterior/evidence/forecast,
preserved parameter uncertainty, incremental conditioning versus full-history
replay, new-law mass, canonical duplication and order invariance, frozen previous
predictions, malformed/unsafe expressions, division/domain/overflow failures,
bounded floating-point powers, resource caps, and all three planner horizons.
Vectorized arithmetic is also compared with the existing scalar interpreter for
every supported operation and function.

These are constructed mechanics tests, not new scientific endpoints or empirical
evidence that an LLM generates useful laws.

## Important limits and next dependency

Selecting expressions or parameter bounds using the same observations used for
fitting is data-dependent model selection. Returned evidence is explicitly
labelled *finite-pool conditional fit, not selection-corrected*. Neither that
evidence nor reduced posterior variance establishes out-of-sample improvement.
Finite prior draws are not a calibrated continuous posterior simply because
their normalization is correct. Parameter integration needs independent
refinement before deployment; fixed-support planning does not anticipate future
structural proposals.

The next useful semantic instrument must compare forecasts sealed before new
observations, on identical fixed targets, from real-history proposals,
history-blind proposals and productive symbolic-search proposals. It must count
proposal and numerical work, constrain proposal priors prospectively, and retain
invalid responses. No method can be certified by selecting the best training
fit or supplying an oracle family name.

This adapter makes that missing dependency implementable. It does not resolve
the failed planning opportunity or open paid requests. A separately justified
source opportunity and prospective semantic/calibration protocol are still
required before any LLM/policy pilot. The complete goal remains unfinished.

Cost: $0. Cluster unused. Automation remains paused. Existing experiment results
and qualified numerical kernels are unchanged.
