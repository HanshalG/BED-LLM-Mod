# First refresh worsens held-out prediction in the saved bank

Retrospective descriptive diagnostic frozen at pushed `e11a8780`, then executed
once in 9.45s. All 32 saved July trees, 256 saved first queries and 5,744 target-query
cases included. No new model call, endpoint opening or inference cost. This does
not rerun or modify the old policies, and changes no old gate or result.

## The relevant finding

| Support after the same first observation | Mean fixed-domain Brier |
|---|---:|
| Initial support, consistency filtered | 0.1875159620 |
| Initial-consistent plus generated answer-conditioned support | 0.1950510101 |
| All speculative first-step proposals shared, then filtered | 0.1861780944 |

Branch refresh is **4.0184% worse** than initial filtering. Its paired per-tree
loss difference is +0.00753505, SD 0.00449749; it wins on 2 trees and loses on 30.
The shared pool is 0.7135% better than initial filtering (18 wins, 14 losses;
paired difference -0.00133787, SD 0.00506792). Shared versus branch-local support
improves 4.5490% (30 wins, 2 losses; difference -0.00887292, SD 0.00455791).

These SDs describe paired tree differences, not uncertainty from independent fresh
experiments. All comparisons are retrospective, with no inferential positive gate.
No arm has an empty support in any evaluated case; the predefined half-probability
fallback was never used. Mean initial/shared pool sizes are 22.1875/216.375.

## What was controlled

Every arm receives the same query and its actual label under each saved target
rule. Every loss uses all integers 0..100, including that query. Target entries keep
their original multiplicity; means weight targets within query, queries within tree,
and trees equally. No favorable query or target is selected.

The shared pool is formed from initial proposals and all first-step generated
proposals for every saved query and both imagined answers BEFORE target labels are
examined. Its creation does not depend on which answer is realized. All extension
hashes/counts, root/branch coverage, observed-label consistency, and exact equality
of saved retained branches to initial-consistent union generated support pass.
Only declared first-step/initial/target fields are materialized; later branches,
validation supports and raw responses are not used.

Four focused tests pass, including an independent hand-built loss calculation,
retention/hash failures, explicit empty prediction and selective streaming. Scoped
lint passes. One test emits an ijson text-stream deprecation warning; the production
bank reader is binary and is not affected. The result is not a new policy replay.

## Interpretation and limits

Answer-consistent extra models are not necessarily useful extra models. Under the
saved uniform-on-unique-support predictor, refresh shifts mass toward worse
predictions on these held-out target rules. That is an observed updater failure,
not merely a numerical planning-rank error. It could reflect the proposal prior,
uniform weighting, support dilution, or mismatch between proposer and target-rule
distribution; this diagnostic does not identify which cause dominates.

The shared pool uses substantially more proposal computation than the initial arm.
It is an availability/control diagnostic, not a compute-matched win or a deployed
policy with free calls. It does demonstrate why a speculative planner must allow
immediate reuse of already-computed models before crediting path-specific discovery.
Its small advantage does not establish that larger pools solve calibration.

Do NOT use the opposite-answer branch as a feedback-sham control here: those
proposals were explicitly filtered for that opposite answer during generation, so
refiltering them against the real answer would produce a tautological failure.
The new prospective shuffled-feedback interface must be tested independently.

Targets are previously generated held-out rule samples, not measured physical
ground truth or proof of an externally calibrated task prior. The old three-query
root-selection result has a different policy and target estimand; this first-step
analysis neither invalidates nor reproduces that result.

## Consequence for the plan

Stop treating a static-support oracle depth gap as sufficient justification for
dynamic LLM planning. It tests one ingredient but not the computational mechanism
in GOAL.md. Conversely, a small fixed-support gap does not mathematically rule out
queries that help a bounded discovery algorithm; our closed configurations remain
closed, not converted into positives by that distinction.

The next LLM-facing dependency is a new, prospectively specified real-history
proposal/update gate whose held-out predictions improve over initial filtering,
history-blind/shuffled proposals and actual symbolic search at stated compute.
An executable structure interface and retained support alone are not enough.
Evaluate proposer and numerical weighting together, then validate the anticipated
transition, rather than launching another depth sweep on unverified refresh.
Use already implemented interpreters/proposers; do not build another small grammar
solely to obtain an attractive classical depth curve.

No closed Number Game interface is reopened, no threshold is relaxed, and no paid
call is authorized by this diagnosis. The full model-discovery/planning/confirmation
goal is still unachieved. Automation remains paused.
