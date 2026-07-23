# UCI Thyroid Workup 26B Named-Proposal Gate Result

The preregistered S0 serving gate and fresh-seed S1 proposal-quality gate **passed
every frozen requirement**. This establishes that non-thinking Gemma 4 26B can
supply useful named branch continuations when the machine fixes candidate roots and
an exact verifier selects the complete two-action policy. It does not yet establish
trajectory-level improvement on fresh patient rows.

| S1 endpoint | Mean | Paired 95% CI | W/T/L |
| --- | ---: | ---: | ---: |
| Matched-random-minus-LLM cost | +0.126507 | [+0.112514, +0.137224] | 30/2/0 |
| Exact-d1-root-minus-LLM cost | +0.129688 | [+0.120230, +0.136457] | 32/0/0 |
| Exact d2 opportunity recovery | 98.353% | [95.060%, 100.000%] | 32/0/0 |

The exact verifier selected the LLM's collection-root policy in 32/32 cells. Gemma
chose `query:tsh` after collection in all 32 cells. One posterior had another
assay as the exact best continuation, producing the only subunit recovery value
(47.3%); all other cells recovered 100% of the exact depth-two opportunity.

S0 accepted all 10 cells on the first response. S1 accepted all 32 cells after
three first-response errors were corrected by the single frozen validation retry.
Across S0/S1 there were 45 physical requests, 93,879 prompt tokens, 5,429 completion
tokens, zero reasoning tokens, zero forced exits, zero rollout/scoring calls, and
`$0` API cost.

An independent audit replayed every history and observation, reconstructed every
named policy and matched-random draw, recomputed 965 unique exact planning subtrees,
and verified every selected slot, root, cost, and control. Fresh-bootstrap intervals
remained positive: `[+0.112880,+0.137170]` versus matched random and
`[+0.120204,+0.136378]` versus the exact-continuation depth-one root. The audit made
zero LLM calls.

The result is specifically about a hybrid non-myopic BED architecture: language
proposes semantic branch continuations, while legal roots and Bayesian scoring stay
machine controlled. The passed gate authorizes only a separately preregistered
paired trajectory confirmation on fresh patient rows.

Artifacts:

- `results/nonmyopic/THYROID_WORKUP_26B_POLICY_PREREGISTRATION.md`
- `results/nonmyopic/thyroid_workup_26b_smoke_20260723/SMOKE.json`
- `results/nonmyopic/thyroid_workup_26b_proposal_gate_20260723/GATE.json`
- `results/nonmyopic/thyroid_workup_26b_proposal_gate_20260723/GATE.md`
- `results/nonmyopic/thyroid_workup_26b_proposal_gate_20260723/AUDIT.json`
- `results/nonmyopic/thyroid_workup_26b_proposal_gate_20260723/AUDIT.md`
