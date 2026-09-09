# Same-objective audit: instrument ready, exact prediction cache absent

The selected structured-V3 initial model is the eight-hypothesis MODEL_FROZEN.json
with SHA256473cf5c883929a2cf8b6d862bebf69e1bb6b6401bea8c7b0edba955e847b7e1d.
It contains semantic map specifications, branches and protocol metadata, not
initial_action_means or initial_heldout prediction arrays. The original selection
script calls simulate_maps to produce those arrays in memory.

The saved DiscoverPhysics FEATURE caches found under results/nonmyopic are:

| Cache | Observation shape | Target shape | SHA256 |
|---|---|---|---|
| dark_matter_opportunity/FEATURES.npz |25x24x2|24x120|130747b3b8bb69987b8db350d08044377cba7a164f287450637ceb773a47fdba|
| dark_matter_asymmetric_opportunity/FEATURES.npz |25x24x2|24x120|daf11e19083b6ab404cebc5cffbdf31ae27825132e229d4dc2a47db41eb9fde4|

These describe older 24-hypothesis experiments and cannot replace the eight-hypothesis
replication model. No risk ranking was computed from the wrong caches. The checked
external/ and tmp/ directories do not contain an obvious DiscoverPhysics checkout;
this is not proof no copy exists elsewhere on the machine.

## Implemented instrument

discoverphysics_myopic_risk computes expected target-feature MSE after ONE
observation, using the same existing Gaussian posterior_batch as the original
experiment. It integrates each initial hypothesis's predictive observation law
with tensor Gauss-Hermite quadrature, averages posterior target variance, and
weights by the exact supplied prior. It has no simulator, endpoint-loader or
model-call invocation. Targets here mean candidate-predicted target features,
not realized hidden-world target labels.

The function supports the original two-coordinate observation and bounded
hypothesis/feature/work sizes. Its output is a quadrature estimate, not an
automatic convergence certificate: the eventual diagnostic must compare multiple
orders before drawing a root-ranking conclusion. Finite differences, posterior
shape and likelihood semantics match the original function; zero-prior hypotheses
do not contribute. Total target MSE is averaged over feature coordinates.

Nine tests pass in .62s, including the prior objective-scope tests. An independent
scalar Gaussian integral agrees to1e-9; uninformative observations preserve prior
risk, separating observations remove it, irrelevant targets give zero, and zero
prior/permutation handling is invariant.

## Next action and limits

Restore or locate the exact upstream33b7fa9df96de9c35744efd181ca7e5a8dd60ad5
source and audit its imports/runtime before executing anything. Freeze a narrow
reconstruction command that reads ONLY the eight public candidate specifications,
compiles them with the original deterministic seeds, and simulates their original
action/forecast inputs. Do not call evaluate_actual, regenerate observed endpoint
worlds, sample new LLM branches, or open private raw reasoning. Retain resulting
candidate arrays with source/model/input hashes, then compare same-objective
one-step risk across all original roots at converged quadrature orders.

That reconstruction is computational work on the saved agent model, not a rerun
of the closed physical experiment. If it cannot be bound exactly, report the
comparison as unavailable, not as evidence for or against horizon value. The
old B-D objective confound remains unresolved by outcomes at this stage.

Previous turn was progress; this turn adds a tested calculator and authoritative
cache mismatch evidence. No model calls, simulator runs or cost. Daily allowance
remaining4.11174654 and balance23.693468061 unchanged. Full goal active/unachieved.
