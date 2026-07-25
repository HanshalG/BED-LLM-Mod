# tau-Knowledge Counterfactual Future-Subtree Replay Result

## Decision

The frozen replay meets the directional intervention-consistency definition
but not the strong definition. Shuffled scores partially recover when judged
against the exact counterfactual outcomes of the future subtrees they saw.
This supports sensitivity to semantic future content, not robust causal use of
the correctly aligned regenerated belief.

## Reproduction

- V3.1 confirmation SHA-256:
  `f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae`.
- Paired future-alignment SHA-256:
  `97d820fb59874370f9ea5a04992398af85503389753a46b9af44fba079b1e736`.
- Every stored source-for-target permutation reproduced exactly.
- All 20 permutations were derangements.
- Every complete future-subtree multiset was preserved.
- Public analysis SHA-256:
  `ef4f970eb53880cc72bb8c1cf7f8dc29cdc4131df5d63e60b268f2cda7af95fe`.
- OpenRouter calls / cost / OatML use: `0 / $0 / none`.

## Results

| Scores | Endpoint world | Root accuracy | Selected root | Selected pair | Optimal follow-ups | Regret |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Aligned | Aligned | `.6529` (121 pairs) | `34` | `29` | `70/100` | `32` |
| Shuffled | Aligned | `.5124` (121 pairs) | `31` | `20` | `67/100` | `35` |
| Shuffled | Counterfactual | `.5660` (106 pairs) | `31` | `24` | `64/100` | `42` |
| Aligned | Counterfactual | `.6226` (106 pairs) | `35` | `28` | `67/100` | `36` |

Relative to evaluating shuffled scores against the original aligned endpoint,
the exact moved-world replay gives:

- root accuracy gain: `+.05364`;
- selected-pair document gain: `+4`;
- task-level root sign-flip `p=.22784`; and
- normalized selected-pair sign-flip `p=.16406`.

## Frozen Classification

The replay passes the `>=.05` accuracy-gain condition, but fails three strong
conditions:

- counterfactual root accuracy `.5660 < .60`;
- task-level root `p=.22784 > .05`; and
- counterfactual-optimal follow-ups `64/100 < 70/100`.

It passes the frozen directional definition because counterfactual accuracy is
at least `.55` and improves over the mismatched endpoint.

## Interpretation

The recovery is the expected signature if GPT-5.4's score responds partly to
the future queries and documents moved into a root. It is not a clean causal
belief-state result. The intervention moves regenerated hypotheses, queries,
and documents together; belief-only shuffling was previously null/adverse.
Moreover, the original aligned score vector performs better than the shuffled
score vector on the counterfactual worlds (`.6226` versus `.5660`) and chooses
four more counterfactual documents (`28` versus `24`).

The calibrated conclusion is:

- semantic full-tree scoring is load-bearing;
- displayed future consequences exert directional influence on scores;
- the effect is heterogeneous and statistically weak at task level; and
- correct path-to-regenerated-belief alignment remains unverified.

This is post hoc analysis on already-open tasks and adds no held-out or
generalization claim.
