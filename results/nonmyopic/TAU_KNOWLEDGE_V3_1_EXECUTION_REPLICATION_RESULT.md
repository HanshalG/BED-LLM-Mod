# tau-Knowledge V3.1 Execution Replication Result

## Decision

The preregistered same-task execution replication failed its primary robustness
gate. The fresh run was mechanically clean, but the non-myopic root-ranking
advantage reversed and end-to-end required-document coverage tied the myopic
control.

This does not erase the original held-out execution. It shows that its positive
ranking and directional endpoint result is not robust to fresh GPT-5.4
generation on the same 20 tasks. The two executions are repeated measurements
of the same tasks, not a 40-task sample.

No third execution is permitted by the preregistration.

## Protocol and Mechanics

The replication used the exact V3.1 policy and the same 20 holdout task IDs,
with a fresh OpenRouter request seed (`24395`). It regenerated every opening,
hypothesis set, query tree, root score, and continuation score. Required
document IDs remained hidden from the model.

The run completed exactly 280 physical requests and 280 HTTP attempts:

- 20 generated openings;
- 20 initial hypothesis/root trees;
- 100 document-conditioned continuation trees;
- 20 isolated myopic root scores;
- 20 full-tree non-myopic root scores; and
- 100 focused receding continuation scores.

There were zero retries, reasoning tokens, forced exits, parse errors, or
runtime failures. Cost was `$2.337425`.

## Replication Results

The fresh execution failed the original V3.1 efficacy gates:

| Component | Replication | Original | Frozen gate |
|---|---:|---:|---:|
| Root comparable pairs | 113 | 121 | >=50 |
| Myopic root accuracy | 0.6283 | 0.5537 | control |
| Non-myopic root accuracy | 0.5575 | 0.7025 | >=0.60 |
| Root accuracy gain | -0.0708 | +0.1488 | >=+0.05 |
| Continuation comparable pairs | 205 | 228 | >=200 |
| Continuation accuracy | 0.7220 | 0.7566 | >=0.60 |
| Oracle-optimal continuations | 74/100 | 80/100 | >=70/100 |
| Mean continuation regret | 0.31 | 0.21 | <=0.30 |
| Selected-root continuation loss | 5 | 3 | <=5 |

Continuation ranking remained useful, but root ranking reversed and mean regret
narrowly missed its gate.

End-to-end required-document coverage was:

| Comparison | Non-myopic | Control | Gain | W/L/T |
|---|---:|---:|---:|---:|
| Myopic receding | 23 | 23 | 0 | 5/4/11 |
| Original joint | 23 | 20 | +3 | 4/2/14 |
| Seeded random | 23 | 23 | 0 | 5/4/11 |

The frozen gates required positive coverage over myopic and at least `+4` over
the joint policy. Both failed.

## Paired and Clustered Analysis

On the replication alone, non-myopic and myopic policies each retrieved a mean
of `1.15` required documents per task. The mean paired difference was `0.00`,
with exact one-sided `p=0.5742` and a task-bootstrap 95% interval
`[-0.50, 0.50]`.

Because both executions use the same task IDs, the pooled analysis averages
within task before inference. It never treats the data as 40 independent tasks.
Across the two executions:

| Metric | Non-myopic | Myopic | Difference | Exact one-sided p | 95% interval |
|---|---:|---:|---:|---:|---:|
| Required-document count | 1.325 | 1.225 | +0.100 | 0.3484 | [-0.275, 0.450] |
| NDCG | 0.2827 | 0.2534 | +0.0293 | 0.2399 | [-0.0458, 0.1068] |
| Reciprocal rank | 0.4913 | 0.4183 | +0.0729 | 0.1489 | [-0.0504, 0.2042] |

The replication-only NDCG and reciprocal-rank comparisons were also null
(`p=0.4404` and `p=0.3398`), and neither rejected after the preregistered Holm
correction.

## Interpretation

The clean mechanics make this a scientific execution-robustness failure, not a
serving failure. The semantic scorer can rank continuations, but fresh
LLM-generated hypotheses, retrieval trees, and scores change which root appears
valuable enough to reverse the first-link result. This is precisely the
path-dependent model noise that makes non-myopic LLM BED fragile: deeper
planning depends on a generated future support that is itself unstable.

The strongest defensible statement is now:

- one held-out V3.1 execution produced significant root and continuation
  ranking with directional endpoint gains;
- a preregistered fresh same-task execution did not reproduce root ranking or
  endpoint superiority;
- task-clustered aggregation across both executions remains directionally
  positive but statistically inconclusive.

The exact Rock Diagnosis results remain the robust evidence that verified
non-myopic recursion can work. The tau-Knowledge result is evidence of a
promising LLM-native mechanism and of its current execution instability, not a
robust positive policy result.

## Artifacts and Budget

- Original confirmation SHA-256:
  `f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae`.
- Replication confirmation SHA-256:
  `627dfe7641dca9fade90890dc90f818da1c6967c08b7878208224cd961fd7f20`.
- Replication analysis SHA-256:
  `14a895286330def28b22085f409af3d086f8e7e959fd691ddfeca038e54ed486`.
- Private raw SHA-256:
  `d5c453f4d657c75c58a2fc21bd637ffcfb628570781e355386f588bb3dad16b2`.

The project ledger is `$89.332866` spent. The authenticated OpenRouter endpoint
reports `$41.051936` remaining, leaving `$16.051936` above the protected `$25`
Monday reserve. OatML was not used.
