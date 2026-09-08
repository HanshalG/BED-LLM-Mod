# Overlap bound valid but fails practical usefulness gate

Frozen implementation/protocol f23e20fa. Artifact SHA256
632d6ab9e8b06aa4b8aa1c7fa69fff6492d406a2c063e1e14a84ed556b49c4e6:
SCILAWS_PARTICLE_OVERLAP_WORKLOAD_20260908.json.
All three first-task child15 cases completed and all24 saved independent risk
values were contained. No reference integral or prior plan rerun.

| History | New/old minimum interval width | Eliminated actions | Seconds/menu |
| --- | --- | --- | --- |
| zero | 0.17224 | 0 | 0.8088 |
| affine | 0.17988 | 0 | 0.9930 |
| quadratic | 0.17547 | 0 | 0.6712 |

The bound materially tightens the noise-only lower bound, but it fails both
the prospective pruning requirement and the0.5s usefulness threshold on every
case. It uses33,554,432 particle-pair terms per menu and a conservative65,290,240
byte temporary-work allowance (below64MiB); it does not store a full pair-target
tensor. These pair terms are real CPU work, not observation evaluations.

A retrospective zero-call check using the banked exact best-action value as
an ideal incumbent also eliminates0 actions in all three cases. Therefore
improving only the linear upper bound cannot make this lower bound prune those
cases. This diagnostic does not authorize oracle information in a real policy.

## Decision

Do not expand the unchanged overlap-bound grid or incorporate this expensive
bound into deployment. Mathematical validity and an82% width reduction are
insufficient when action gaps are smaller and bound evaluation itself consumes
substantial decision time. Keep the implementation as a tested diagnostic,
with no production default change. No threshold rescue.

The remaining credible computational direction is to amortize full likelihood/
moment integrations across actions or exploit the target-feature structure,
with measured whole-menu throughput and independent numerical equivalence.
Another success on a tiny bound or a small scalar speedup would not establish
complete decision feasibility. Test one bounded all-action operator before any
new full refinement grid. Full outer error, source calibration and the actual
LLM/non-myopic contribution remain unproven. Do not buy proposer calls yet.

Four tests passed0.56s and scoped lint passed. Process exited, source/model
calls0, cost0, authenticated account and London Sept8 ledger unchanged.
Automation remains paused; the full research goal remains unfinished.
