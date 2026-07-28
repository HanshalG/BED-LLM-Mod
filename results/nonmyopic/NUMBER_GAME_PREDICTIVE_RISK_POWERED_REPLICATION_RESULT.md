# Number Game Predictive-Risk Powered Replication Result

Date: 2026-07-28

Protocol:
`NUMBER_GAME_PREDICTIVE_RISK_POWERED_REPLICATION_PREREGISTRATION.md`

Run:
`results/nonmyopic/number_game_predictive_risk_powered_replication/number-game-predictive-risk-powered-replication-20260728T143000Z`

## Verdict

**All preregistered gates pass.**

This is the project's LLM-native non-myopic result. The LLM is not a strategy
ornament: Gemini generates the open-ended executable hypothesis support and
its branch-conditioned successor supports. Predictive-risk BED chooses the
first query by simulating those future belief states. GPT-5.4 independently
generates the hidden target concepts.

All 32 planning trees and 32 target supports are fresh relative to development
and the earlier eight-tree replication. Primary inference uses only these 32
trees.

## Primary Results

| Baseline | Candidate Brier | Baseline Brier | Relative gain | Brier wins | Whole-tree 95% difference CI |
|---|---:|---:|---:|---:|---:|
| myopic EIG | 0.18359 | 0.23086 | 20.48% | 30/32 | [-0.06281, -0.03333] |
| fixed-support depth two | 0.18359 | 0.23355 | 21.39% | 31/32 | [-0.06487, -0.03691] |
| exact uniform random root | 0.18359 | 0.20779 | 11.65% | 30/32 | [-0.02951, -0.01877] |
| seeded PTS | 0.18359 | 0.20437 | 10.17% | 27/32 | [-0.02705, -0.01431] |

Predictive-risk BED selects a different first query from both myopic EIG and
fixed-support depth two on all 32 trees.

Best-rule Hamming error is:

- `0.08943` versus myopic `0.13774`, a `35.07%` reduction with whole-tree
  difference CI `[-0.05827, -0.03822]`;
- `0.08943` versus fixed-depth-two `0.13987`, a `36.06%` reduction; and
- `0.08943` versus PTS `0.10497`, a `14.81%` reduction with CI
  `[-0.02205, -0.00925]`.

Mean exact-extension coverage rises by `10.24` percentage points versus
myopic, `11.31` points versus fixed depth two, `4.87` points versus random, and
`3.98` points versus PTS.

For target extensions absent from each planning support, candidate-minus-
myopic mean differences are `-0.05499` Brier and `-0.04927` Hamming. The
candidate wins 28/32 novel-target tree means on Brier and 30/32 on Hamming.

## Why This Is Non-Myopic and LLM-Native

All policies share:

- the same deterministic Number Game likelihood;
- the same initial particle support within a tree;
- the same branch-conditioned generated support after their realized root;
- the same greedy-EIG second-query rule; and
- the same independently generated endpoint targets.

They differ only in first-query selection. Myopic EIG values the current
support. Fixed-depth-two plans on that support without regeneration.
Predictive-risk BED evaluates the posterior predictive loss that remains after
the LLM has regenerated a branch support and a second adaptive query has been
answered. Classical exhaustive search cannot reproduce this score without the
LLM's counterfactual proposal process.

The earlier eight-tree V2 remains a formal null against its own 5% PTS
magnitude gate. This separately frozen powered study required a prospective 2%
PTS effect and passes at 10.17%.

## Accounting

- Successful responses: `576`
- HTTP attempts: `578`
- Explicit zero-cost provider-error retries: `2`
- Reasoning tokens / forced exits: `0 / 0`
- Cost: `$1.4966614`
- Result SHA-256:
  `a48ccef1637bbab3ad939a4785ec5128e33d09ecba7714775dc1b3a47f071ba9`
- Tree artifact SHA-256:
  `52c2c6cf8126d9f78fb794e6f6e78a651fccebbf9084210ceeebac7e936ad282`
- Private raw-response SHA-256:
  `bc5047e2c763bc59c094486ff752eef4fd74213a4f82066fcc0b75f9f784893d`
