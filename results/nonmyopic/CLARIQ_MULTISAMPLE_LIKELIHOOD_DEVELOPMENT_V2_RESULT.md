# ClariQ Multisample Likelihood Development V2 Result

## Decision

V2 fails its preregistered conjunction because one of three topics misses the
single-sample modal-selection stability gate. There is no V2 rerun, sample-count
change, topic substitution, or threshold repair.

The scientific first-link result is nevertheless strong enough to motivate one
separately preregistered holdout test: every efficacy, ranking, endpoint, serving,
and cost gate passed.

## Serving

- Model: `openai/gpt-5.4`, non-reasoning.
- Requests / HTTP attempts: `205 / 205`.
- Retries / reasoning tokens / forced exits: `0 / 0 / 0`.
- Exact parsed maps: `205 / 205`.
- Prompt / completion tokens: `41,180 / 1,229`.
- Cost: `$0.121385`.

## First-Link Results

| Topic | Myopic root | Depth-two root | Myopic tail | Depth-two tail | Gain |
|---|---|---|---:|---:|---:|
| `136` | `Q01414` | `Q00696` | `.161657` | `.231354` | `+.069697` |
| `125` | `Q03447` | `Q00773` | `.262187` | `.270748` | `+.008561` |
| `149` | `Q01473` | `Q01473` | `.796251` | `.796251` | `0` |

Depth two changed two roots, won both changed decisions, lost none, and tied
the third. Mean external oracle-tail gain over myopic was `+.026086` NDCG@20;
mean gain over seeded random was `+.032847`.

Across all 41 external roots:

- depth-two score versus oracle-tail Spearman: `.440748`;
- myopic score versus oracle-tail Spearman: `.358276`.

Thus the non-myopic LLM-likelihood score improves both selected first actions
and global root-value fidelity on these development topics.

## Failed Stability Gate

Five one-sample depth-two planners had to share a modal root at least three
times on every topic:

- topic `136`: `Q00696` selected `3/5`;
- topic `125`: `Q00852` selected `3/5`;
- topic `149`: no root reached `3/5` (`Q00697` and `Q00106` each `2/5`).

Topic `149` is also the only topic where full five-sample depth two equals
myopic. The failed single-sample gate therefore does not create either positive
endpoint gain. Still, the preregistered conjunction fails.

The disclosed post hoc stability analysis shows:

- topic `125` full root agrees in `5/5` leave-one-out fits and `77.3%` of
  five-draw bootstraps;
- topic `136` agrees in `3/5` leave-one-out fits and `49.5%` of bootstraps;
- unchanged topic `149` has an exact full-score tie among leading roots,
  agrees in `4/5` leave-one-out fits, and selects its tie-broken full root in
  `43.4%` of bootstraps.

This motivates evaluating stability only where depth two changes the deployed
root in a prospective holdout protocol. It is development-informed and must be
disclosed as such.

## Interpretation

This is the strongest fresh LLM-native first-link evidence after tau:

- actions and answer transitions are human-authored;
- the endpoint is official retrieval NDCG;
- myopic and depth two use identical questions and identical model calls;
- the LLM is load-bearing because its semantic `Y/N/U` likelihoods define the
  belief transition and both acquisition scores; and
- non-myopia improves the exact external future value of the chosen root.

It remains a three-topic development result and failed one frozen stability
condition. It is not yet a headline claim.

## Artifacts and Budget

- Public development artifact:
  `results/nonmyopic/clariq_multisample_likelihood_v2_development/clariq-multisample-likelihood-v2-development-20260725T163350Z/DEVELOPMENT.json`
- Public SHA-256:
  `6daf922d9bfca5c2c28a843fddde58970b13219aa1981ea1836f033cc069c4fc`
- Private raw SHA-256:
  `e459fb387149d3f637f1b61a1bb7011f1c2bcc98b4bea10fa8aa54830cd9fbd4`
- Post hoc stability:
  `results/nonmyopic/clariq_multisample_likelihood_v2_development/STABILITY.json`
- Stability SHA-256:
  `4f15a02a306c17271e833d0aa350564f78e9d2735da08ee1747040f6df126023`
- Project ledger after run: `$89.454251`.
- Remaining local pre-Monday allowance: `$11.689054`.
- OatML use: none.
