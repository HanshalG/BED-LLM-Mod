# Number Game Size-Matched Support Development Result

Date: 2026-07-28

This is a post-hoc, zero-call mechanism audit over three independently
collected 32-tree datasets. It strengthens interpretation but is not a new
preregistered confirmation.

## Question

The preregistered global-pool control received many more rules than a realized
root-conditioned branch. Its loss could therefore reflect either harmful
support volume or the absence of path-specific semantic focusing.

For every tree, root, and answer, this audit:

1. deduplicates every generated rule into one global static pool;
2. filters that pool by the contemplated answer;
3. samples without replacement to exactly match the corresponding realized
   branch size;
4. repeats this for 32 deterministic static-support samples;
5. selects a root under each sampled static world; and
6. averages their held-out deployment endpoints.

All endpoint policies use the original realized branches. Only first-root
selection differs. The fast vectorized selector was checked against the
reference exact selector on every tree.

## Primary 32-Sample Result

| Fresh dataset | Candidate Brier | Size-matched Brier | Relative gain | Brier wins | Whole-tree 95% difference CI |
|---|---:|---:|---:|---:|---:|
| Gemini planner / GPT targets, powered | 0.18359 | 0.20303 | 9.58% | 29/32 | [-0.02507, -0.01376] |
| GPT-mini planner / Gemini targets | 0.18898 | 0.21387 | 11.64% | 29/32 | [-0.03300, -0.01738] |
| Gemini planner / GPT targets, confirmation | 0.18381 | 0.20084 | 8.48% | 30/32 | [-0.02135, -0.01309] |
| Combined descriptive | 0.18546 | 0.20591 | 9.93% | 88/96 | [-0.02402, -0.01701] |

Combined best-rule Hamming falls by `14.61%`, with whole-tree difference
interval `[-0.02089, -0.01317]` and 78/96 wins. Exact-extension coverage rises
by `4.01` percentage points. On target extensions absent from initial support,
candidate-minus-control differences are `-0.01960` Brier and `-0.01650`
Hamming, with 79/96 and 74/96 wins.

The candidate differs from the modal size-matched root on 71/96 trees. Across
the 32 static samples, the candidate root is selected only `19.11%` of the
time on average. Thus the endpoint gain is not produced by nearly identical
root choices.

## Monte Carlo Sensitivity

At 128 deterministic samples per tree, combined Brier gain is `9.64%`
(`[-0.02312, -0.01656]`, 88/96 wins), Hamming gain is `14.39%`
(`[-0.02049, -0.01294]`, 78/96 wins), and coverage gain is `4.07` points.
Every per-dataset Brier and Hamming interval remains wholly negative.

## Interpretation

Root-conditioned generation is not winning because it receives more
hypotheses. It wins against a static global proposal process with exactly the
same branch sizes, including on novel target extensions and after swapping
planning and target model families. The remaining advantage is semantic
focusing: which executable concepts the LLM proposes after a particular
query, not merely support count.

The audit remains post-hoc and uses the restricted Number Game grammar. A new
paid confirmation was not launched because the remaining provider balance
cannot support a meaningfully powered fourth fresh dataset.

Primary result SHA-256:
`93b086b830a2d5a3242ad410837e63176e858c3444a6389adf1bc131d6332ac7`

128-sample sensitivity SHA-256:
`1997ce2f90c0f96429db1605c4707598b0edc7b4c40ba1b572032d5444399468`
