# MovieLens Depth-2 Semantic Policy v8 Diagnosis

Date: 2026-07-24

Status: exploratory analysis of the frozen paid tree; zero additional model calls.

The preregistered v8 policy gate failed. This analysis uses all four first-query
choices for each of the four enrolled users to localize why.

## Fidelity Chain

| Link across 16 first-query choices | Spearman |
|---|---:|
| Depth-2 expected value vs realized terminal entropy | +0.1618 |
| Realized terminal entropy vs realized held-out NLL | +0.2941 |
| Depth-2 expected value vs realized held-out NLL | **-0.5265** |
| Depth-1 expected value vs realized held-out NLL | **-0.5500** |

The main break is the first link: averaging hypothetical semantic transitions does
not predict the terminal belief selected by the realized outcomes. Depth 2's mean
oracle regret was `.0663`, versus `.0347` for depth 1 and `.0656` for immediate EIG.

User 933 is the clearest example. Depth 2 predicted *The Fugitive* as best with
expected terminal entropy `1.4373`, but its realized terminal entropy was `1.5792`
and held-out NLL `1.7595`. The depth-1 choice finished at `1.5463`.

## Closed Repairs

All repairs below were evaluated post hoc on the same frozen tree and authorize no
claim or fresh run:

- Rollout mutual information, which replaces the current-belief entropy assumption
  with each action's branch-weighted marginal, had Spearman `-0.5471`, mean oracle
  regret `.0960`, and selected no oracle action.
- Fixed risk penalties `mean + 0.5/1/2 SD` remained negatively correlated. A
  worst-branch rule reduced mean regret to `.0485` but still had Spearman `-0.5353`.
- A global temperature fit on the 44 disjoint non-enrolled screen users was
  `T=1.1362`; it improved calibration NLL only `1.4822 -> 1.4804` and left planner
  Spearman `-0.4706`.
- A six-profile Bayesian lineage weighting retained profile identities across both
  transitions, but had Spearman `-0.1118`, mean regret `.1016`, and zero oracle
  selections.

The branch-weighted terminal prediction also drifted substantially from the initial
prediction (mean per-movie L1 ranges roughly `.12-.35`). Repeated LLM regeneration is
therefore not a coherent Bayesian transition, and simple calibration or risk
regularization does not make it one.

## Decision

Close this MovieLens depth-two entropy planner. V7 remains a valid positive one-step
mechanism-ranking result, but v8 shows that composing the transition twice destroys
value fidelity. Do not spend on a larger cohort, more rollouts, temperature tuning,
MI scoring, or lineage weighting for this exact tree. The next headline attempt must
change the belief representation or environment rather than tune this null.
