# MuSiQue 4-Hop Branching Causal Smoke Result

Date: 2026-07-25

## Outcome

**The exact sixteen-call smoke passed all serving and belief-causality checks
but failed the conjunctive scientific gate.** The exact generic-utility
interface is closed. The other 38 development and all 240 holdout endpoints
remain sealed.

Public artifact:

`results/nonmyopic/musique_branching_causal_smoke/musique-branching-causal-smoke-20260725T043430Z/SMOKE.json`

## Mechanics

| Metric | Result |
|---|---:|
| Physical requests / HTTP attempts | 16 / 16 |
| Transport retries | 0 |
| Reasoning tokens / forced exits | 0 / 0 |
| Parsed responses | 16 / 16 |
| Refreshed states differ from initial | 12 / 12 |
| Aligned score vectors vary | 12 / 12 |
| Aligned vs shuffled vectors differ | 12 / 12 |
| Aligned vs initial vectors differ | 12 / 12 |
| Deep-root aligned continuation retrieves deep child | 2 / 2 tasks |
| Prompt / completion tokens | 31,411 / 9,664 |
| Cost | $0.2234875 |

The LLM generated at least three distinct first retrievals per task and at
least five distinct refreshed states per task. The state-only scorer saw no
question, first query, first result, answer, or annotations.

## Policy Result

| Task | Generated deep/shallow roots | Myopic prefix | Model-aware prefix | Model root |
|---|---:|---:|---:|---|
| `405751...55609` | yes / yes | 1 | 1 | other support |
| `360282...54023` | yes / no | 2 | 2 | deep root |

Model-aware was never worse than myopic, fixed-support, or shuffled-belief d2,
and selected a deep root once. It did not strictly beat myopic, and the second
task failed the required generated-root coverage because no query retrieved
the shallow support.

On task 1, three of six generated roots retrieved the annotated deep root and
two retrieved the shallow root. Every deep-root branch generated a follow-up
that retrieved the deep child. Nevertheless:

- myopic, fixed-support, and model-aware all selected the later support
  `History of taxation in the United States`, then the Sixteenth Amendment,
  for connected-prefix length 1;
- the seeded random policy selected a deep-root query (`David Vladeck`) and
  its generated child (`Separation of powers...`), reaching exact prefix 2;
- model-aware root totals for the three deep candidates were 159, 156, and
  165, below 187 for the selected later support.

The six model-aware root scores had Spearman `-.878` with each root's exact
best reachable prefix on task 1 and `+.683` on task 2; pooled correlation was
only `+.073`. Immediate scores had the same within-task correlations, so the
generic continuation scores did not repair first-action ranking.

## Interpretation

This is a stronger causal negative than Hotpot. The higher-hop environment
prevents commutative evidence recovery, the LLM generates the correct
observation-conditioned deep-child queries, and the aligned belief changes
every score vector. The remaining failure is specifically the first-action
objective: arbitrary 0--100 immediate and continuation usefulness scores do
not represent comparable connected-chain progress.

No response is rerun, no task is replaced, and no score rescaling or weight is
chosen after seeing the endpoint. A scientifically distinct development
successor could replace generic utilities with a prospectively frozen,
discrete terminal dependency-progress objective, using new tasks. It would
need its own preregistration and could not rescue this V1 result.

