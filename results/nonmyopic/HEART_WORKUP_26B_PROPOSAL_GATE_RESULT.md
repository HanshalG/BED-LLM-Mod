# Cleveland Heart Workup 26B Proposal Gate Result

The preregistered S1 exact proposal-quality gate **failed all four endpoints** in job
`106374`. The Heart LLM-policy line stops here; no formal trajectory comparison is
authorized.

| Endpoint | Mean | Paired 95% CI | W/T/L |
| --- | ---: | ---: | ---: |
| Matched random - LLM cost | -0.007814 | [-0.033967, +0.023341] | 6/9/17 |
| Strong d1 - LLM cost, unworked | -0.043493 | [-0.070362, -0.008328] | 1/0/15 |
| D2-opportunity recovery, unworked | -1.562626 | [-2.760557, -0.674464] | 1/0/15 |

The exact verifier selected the workup-root policy on only 1/16 unworked d2
opportunities (6.25%), versus the frozen 75% threshold. Every one of these 16 cells
strictly favored workup under exhaustive d2.

## Failure Mechanism

Serving was perfect: all 32 cells were accepted first attempt, all policies were
complete and legal, and the exact verifier and controls ran without LLM calls. The
failure came from a stable positional proposal heuristic:

- For the workup root, Gemma chose the first local menu item in 15/16 cells. That
  produced `query:age` nine times and `query:sex` six times. The exact post-workup
  continuation was `major-vessels` in 12 cells, `thal` in three, and `st-slope` in
  one. The only workup-root policy selected by the verifier was the one cell where
  Gemma chose `thal`.
- For every branch of every ordinary-query root, Gemma chose local index zero,
  `order:clinical-workup` (120/120 branches). This yields a query followed by a
  zero-information setup action inside the two-step scoring horizon.
- Consequently the exact verifier selected fixed-root slot one, usually the myopic
  query, on 15/16 unworked cells. The LLM policies were significantly worse than the
  strong d1 root with exact continuation.
- In already-worked cells the model frequently selected clinically relevant detailed
  tests (`major-vessels` 58 and `thal` 45 branch follow-ups), but pooled performance
  still did not beat matched random.

An independent replay reconstructed all 32 cells and reproduced every LLM/random
cost, selected root, bootstrap comparison, and failed gate within `1e-12`.

Usage was 32 requests, 53,951 prompt tokens, 1,229 completion tokens, zero reasoning
tokens, zero forced exits, zero rollout/scoring LLM calls, and `$0` API cost.

Artifacts:

- `results/nonmyopic/HEART_WORKUP_26B_POLICY_PREREGISTRATION.md`
- `results/nonmyopic/heart_workup_26b_proposal_gate_20260723/GATE.json`
- `results/nonmyopic/heart_workup_26b_proposal_gate_20260723/GATE.md`
- `results/nonmyopic/heart_workup_26b_proposal_gate_audit_20260723/AUDIT.json`
- `results/nonmyopic/heart_workup_26b_proposal_gate_audit_20260723/AUDIT.md`

Interpretation: the exact Cleveland environment contains a real non-myopic advantage,
but this compact indexed Gemma 4 26B proposer does not recover it. This is a proposal-
ranking failure under the frozen interface, not evidence against the exact d2 policy.
