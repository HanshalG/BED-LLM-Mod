# Development simulator contracts

The frozen eight-task development panel passed restricted state inspection at
dataset revision `15b93258fabc8f1785c5a5ffd4d3e51fc15dd860`. Result SHA256:
`f6f5a18208857c62b3fe5a84532aecbce330ef17a0bba521c44da08f8a462226`.
Exact saved development order and complete coverage were independently checked.
No holdout or guarded-sibling state was downloaded. No hidden formula was
executed, new measurement generated, policy endpoint opened, or model called.

| Task | Inputs | Residual space | Source row budget |
|---|---:|---|---:|
| Baseball win percentage | 3 | Linear | 59000 |
| Bird flight speed | 2 | Linear | 2620 |
| Lake thermocline depth | 1 | Linear | 24460 |
| Battery capacity fade | 1 | Linear | 12720 |
| Mars crater frequency | 1 | Log | 2000 |
| Spirometry FEV1 | 2 | Linear | 72500 |
| Volcanic column height | 1 | Log | 2600 |
| Wind turbine power | 1 | Linear | 6809600 |

All eight have finite, increasing input bounds and nonzero noise scale. The
source budgets are availability limits, not permission to spend those rows.
The saved contract contains exact input names and bounds, not response values.

## Scientific implications

This is a source-feasibility pass, not a horizon-opportunity pass. The upstream
noise is local empirical residual resampling, not automatically Gaussian or
zero-mean. Six linear and two log-space tasks require an explicitly justified
agent observation model. Hidden residual atoms and scale magnitudes remain
excluded from agent inference. A privileged true-world reference must be
labelled separately from a deployable prior-based planner.

Several input supports span orders of magnitude; a common linear grid would
implicitly overweight large values. Input transforms, action menus and fixed
target weights must therefore be chosen from public contracts before outcomes,
not selected retrospectively for a depth gain. Box-valid combinations of multiple
inputs also need a declared interpretation: this benchmark is a synthetic
simulator experiment, not evidence that every combination is physically feasible.

Next: freeze a bounded eight-task opportunity protocol specifying initial
evidence, structure/parameter prior, inferred noise, target loss and weights,
ordinary receding h1/h2/h3, open-loop and compute-matched myopic/random controls,
paired noise streams, numerical refinement and runtime caps. Resolve source
attribution/usage obligations before measurement generation. No task substitution,
noise reduction or target reweighting after a null. LLM proposal and calibration
gates follow only if the numerical opportunity is real; none is authorized here.

## Reader and limitations

States are LFS-hash-verified and stored outside git in an owner-only directory.
The child reader substitutes an inert bootstrap object and admits only specific
numeric serialization classes. It returns an allowlisted projection and rejects
object arrays. Hidden state fields are materialized privately during loading;
this is not a claim that the raw state was never opened. The reader neither
imports the upstream simulator nor evaluates its hidden formula.

This is not a formal malicious-pickle security proof or aggregate memory sandbox.
The 20 MiB file/per-array limits do not bound total decompressed memory. Tested
versions are joblib 1.5.1 and NumPy 1.26.4. Four focused tests pass, covering
compressed/uncompressed projections, rejected executable globals and rejected
object arrays. Lint and independent result order/coverage checks pass.

Authenticated credits/usage/balance remain 245/220.376693994/24.623306006;
the London daily ledger validates zero spend. Automation remains paused.
The full research goal and a positive LLM-native non-myopic result remain unfinished.
