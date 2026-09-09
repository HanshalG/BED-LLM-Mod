# NeuronBench's saved levels are not ordinary horizons

Reassessment after the frozen Black source diagnostic, not a new experiment.
No source outcome, model response, or old gate changed. The August15 deterministic
NeuronBench formulation remains closed. This correction prevents its large mean
ladder from being used as evidence for the requested ordinary-horizon opportunity.

## Executable finding

Inspected environments/neuronbench_compose/mechanics.py SHA256
f9f32c59b9d0cc7687874b6b83be9efd95ad5fe717dd0f58723c8108539b4e92.
CompositionalPlanner.action_value lines269-274 calls policy_value with remaining-1
and level-1. policy_value lines300-319 then executes that policy through all
remaining actions while keeping its level constant. Thus level2 at a four-query
root evaluates a first query plus the entire three-query level1 continuation,
not a two-query truncated lookahead. Level3 evaluates the full lower-level policy
in turn. The class docstring correctly calls this finite-budget policy improvement.

New regression uses the real saved planner with a deterministic counting bank:
leaf risk10 minus number of observations, four available queries. Level1 action
value is9; level2 is6, whereas an actual two-observation terminal value is8.
At fixed level2, changing execution budget3 to4 changes evaluated risk7 to6.
Both tests pass in .17s. They characterize code semantics, not scientific efficacy.
Test SHA256db024c18b055a7eef1fd8624c35d36f061b1df1bf53fbf91d44fd9f1953a6317.

The separate branch model also averages over bank.truth_indices, with posterior
weights carried separately from represented proposal support. This is an oracle
population model, not access to one realized truth, but cannot simply be copied
into an LLM experiment whose predictor sees less. The September chemistry critique
therefore applies here too; it was not enough to correct chemistry's terminology.

## Consequences

The old1.137226/.334017/.156647 expected-MSE ladder is still its valid recorded
estimand, subject to its existing failed gates. It is not a measured h1/h2/h3 curve.
Calling its 70.63%/53.10% drops evidence of large ordinary-horizon headroom would
overstate what was tested. Do not rerun the closed study, change its paired-win
gate, or reclassify its null as a positive.

Our strongest banked ordinary-horizon evidence remains the September Number Game
initial-belief audit: .0700803/.0644678/.0631955 exact model-relative Brier, with a
1.97% last increment and its original null intact. That evidence has real contingent
structure but does not establish calibrated held-out LLM discovery. Its first-refresh
audit independently shows deterioration under branch-local uniform support. These
are different estimands; neither can be used to fill the other's missing proof.

## Literature check and architecture decision

MDA v4 separates LLM structural proposals from numerical inference and one-step
VoI. Its NeuronBench section enumerates a discrete experiment menu; its proposer
specifies parameterized channel properties. The paper does not supply a controlled
successive ordinary-horizon result. This supports the architectural separation,
not reviving our failed ladder as a replication of a non-myopic result.
[Primary paper, sections E.3.2 and H.4.1](https://arxiv.org/html/2608.09696v4).

Stop the pattern of choosing a new familiar benchmark and engineering its runtime
before establishing the experimental uncertainty it offers. The Black domain
demonstrated this failure mode directly. DiscoveryWorld's banked deterministic
wrapper fixes tested replay, but its missing scientific-law prior and expensive
action interface remain; its name alone is not a successor design.

The next useful retrospective calculation is an ordinary truncated-horizon audit
on the already-opened NeuronBench response bank, using one explicitly shared
prediction model and a full-information-access predictor control. That calculation
has not been done by the existing ladder and would answer whether an ordinary gap
exists there at all. Freeze scope and runtime caps first; retain all15worlds and
all9actions. It must be labelled an oracle source diagnostic, not a reopening of
the old gate or authority for a paid study. A null should stop further neuron
runtime/model work; a gap would still need a genuinely new, prospectively frozen
LLM proposal/transition-calibration study before a policy claim. Do not create
another source registry solely to make the gap larger.

## Accounting

Previous turn progress; current new executable evidence changes environment
prioritization and corrects an interpretation. No calls or cost. Authenticated
balance23.693468061, London-day conservative remaining4.11174654 unchanged.
Old reports and endpoint banks untouched. Full goal active/unachieved.
