# tau-Knowledge Refreshed-Belief Alignment Ablation Result

## Decision

The serving smoke passed, and the exact 120-call confirmation completed without
an interface failure. All three frozen mechanism criteria failed. The current
tau result therefore supports load-bearing semantic LLM scoring, but it does
not isolate correct branch-to-refreshed-belief alignment as load-bearing.

## Intervention

Within each task, a fixed seed-24343 derangement moved every branch's
`refreshed_information_need_hypotheses` to a different branch. The opening,
initial beliefs, all queries, all retrieved documents, endpoints, model, prompts,
and parsers were unchanged. All 20 permutations were derangements, every
per-task belief multiset was preserved, and canonical non-belief hashes matched.

Before model use, the 100 refreshed states were verified to be substantive:
all tasks had five distinct states, no within-task pair was identical, and mean
within-task token Jaccard was `.2879`.

## Serving

The two-task mechanics smoke passed:

- exact physical requests: 12;
- reasoning tokens and forced exits: 0;
- root and focused schemas: all complete;
- cost: `$0.127205`; and
- all intervention invariants: passed.

No smoke efficacy threshold was imposed because degradation was the estimand.

## Confirmation

The confirmation completed exact 120/120 calls with zero reasoning tokens,
zero forced exits, and cost `$1.222875`.

| Metric | Full alignment | Shuffled alignment | Full minus shuffled | Frozen requirement |
|---|---:|---:|---:|---:|
| Root pairwise accuracy | .7025 | .7273 | -.0248 | >= .05 |
| Continuation pairwise accuracy | .7566 | .7412 | .0154 | >= .05 |
| Endpoint documents | 30 | 34 | -4 | >= 3 |

The one-sided task-level sign-flip values for a full-alignment advantage were
`.6191`, `.2551`, and `.9453`, respectively. These are descriptive because the
trees and original full-alignment endpoints were already open.

The shuffled run itself passed every original V3.1 efficacy gate: it selected
83/100 oracle-optimal continuations, had mean regret `.17`, and retained all but
two documents available under its selected roots. Relative to the controls
recomputed with the shuffled focused scorer, it gained 10 documents over myopic,
6 over the original joint root/followup choice, and 12 over seeded random.

## Decision Drift

The intervention was not behaviorally inert:

- root argmax agreed with the original full-alignment run on 9/20 tasks;
- focused followup argmax agreed on 69/100 roots;
- no root score vector and only 1/100 focused score vectors were exactly
  identical; and
- shuffled endpoint outcomes beat, tied, and lost to the original on 5, 13, and
  2 tasks, for total `+4`.

The original and shuffled scores were obtained in separate OpenRouter runs, so
some score drift may be backend nondeterminism rather than the belief
intervention. This prevents claiming that wrong alignment improves performance.
It does not rescue the preregistered causal claim: no measured endpoint worsened
by the required amount, and all full-alignment-advantage tests were null.

## Interpretation

The exact retrieved evidence and the semantic document scorer are sufficient to
recover the successful policy behavior even when refreshed beliefs are attached
to the wrong within-task branches. The strongest supported description is
therefore:

- the LLM generates semantic query trees;
- an LLM semantic scorer is load-bearing relative to deterministic retrieval
  heuristics; and
- non-myopic visibility plus receding continuation selection improves ranking.

The stronger statement that the scorer succeeds because it tracks the LLM's
correct path-dependent refreshed belief state is not supported. No paired replay,
new permutation, threshold change, or prompt rescue was run after this null.

## Artifacts

- Preregistration:
  `results/nonmyopic/TAU_KNOWLEDGE_BELIEF_ALIGNMENT_ABLATION_PREREGISTRATION.md`
- Smoke:
  `results/nonmyopic/tau_knowledge_belief_alignment_ablation_smoke/tau-knowledge-belief-alignment-smoke-20260725T020422Z/SERVING_SMOKE.json`
- Confirmation:
  `results/nonmyopic/tau_knowledge_belief_alignment_ablation_confirmation/tau-knowledge-belief-alignment-confirmation-20260725T020439Z/CONFIRMATION.json`
- Private raw responses: stored outside git.
