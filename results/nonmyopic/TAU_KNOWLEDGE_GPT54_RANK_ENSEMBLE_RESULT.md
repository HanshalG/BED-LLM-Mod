# tau-Knowledge GPT-5.4 Rank-Ensemble Result

## Decision

The frozen same-task rank-ensemble confirmation failed one endpoint gate. The
ensemble reproduced strong ranking and a net `+4` document advantage over its
compute-matched myopic ensemble, but lost on `4/20` tasks versus the allowed
maximum of `2/20`. No larger ensemble, normalization change, or threshold
repair follows.

## Mechanics

- Fresh scorer replicates: `3/3`.
- Physical requests: exact `420/420`.
- Reasoning tokens: `0`.
- Forced exits or malformed responses: `0`.
- Tree, query, retrieval, belief, or endpoint regeneration: `0`.
- Adapter-attributed cost: `$1.7954145`, below the `$6.75` cap.
- Live balance afterward: `$46.563267126`, leaving `$21.563267126` above
  the protected reserve.

The three component runs had standalone non-myopic/myopic endpoints of
`32/26`, `30/29`, and `31/26`. Individual endpoint gates were descriptive;
the preregistered policy was the single three-call rank ensemble.

## Confirmation

| Metric | Development ensemble | Fresh execution ensemble | Gate |
|---|---:|---:|---:|
| Root pairwise accuracy | .6736 | .6901 | >=.60 |
| Root gain over myopic | +.0909 | +.1198 | >=.05 |
| Focused accuracy | .7610 | .7719 | >=.60 |
| Focused optimal rate | .83 | .83 | >=.70 |
| Focused mean regret | .17 | .17 | <=.30 |
| Non-myopic endpoint | 30 | 30 | >=25 |
| Myopic endpoint | 26 | 26 | control |
| Net advantage | +4 | +4 | >=+4 |
| Wins / losses / ties | 5 / 2 / 13 | 6 / 4 / 10 | wins>=4, losses<=2 |

The confirmation ensemble also gained `+5` documents over the joint
continuation control and `+8` over frozen random. It selected the same
non-myopic root as the development ensemble on `.75` of tasks and the same
focused continuation on `.84` of all roots, passing the `.60/.75`
reproducibility gates.

Every original and strongest-nonsemantic gate passed except
`end_to_end_losses_to_myopic_at_most_2`.

## Loss Structure

The six winning tasks contributed `+8` documents and the four losing tasks
contributed `-4`, giving net `+4`; ten tasks tied. The four losses were
one-document reversals on `task_028`, `task_029`, `task_089`, and `task_017`.

Thus score-scale normalization did exactly what its mechanism predicted:
ranking accuracy, aggregate coverage, and cross-block decisions were stable.
It did not make non-myopic selection uniformly safer than the equally
ensembled myopic policy.

## Interpretation

This is stronger reproducibility evidence for the semantic ranking links and
for the net directional endpoint gain, but it is not an all-gates policy
confirmation. The frozen conservative loss cap prevents relabeling a
`6/4/10` task pattern as a robust win.

The method was developed on the first three executions and confirmed on fresh
model executions of the same open tasks. It therefore addresses provider
execution noise only; it adds no new-task generalization, cross-model transfer,
or evidence that correctly aligned path-dependent beliefs are causal.

## Artifacts

- Preregistration:
  `results/nonmyopic/TAU_KNOWLEDGE_GPT54_RANK_ENSEMBLE_PREREGISTRATION.md`
- Aggregate:
  `results/nonmyopic/TAU_KNOWLEDGE_GPT54_RANK_ENSEMBLE_ANALYSIS.json`
- Three fresh parsed scorer artifacts:
  `results/nonmyopic/tau_knowledge_gpt54_rank_ensemble_confirmation/`
- Private raw responses: stored outside git.
