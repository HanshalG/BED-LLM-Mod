# Mushroom Feature Acquisition 26B Proposal Gate Result

## Decision

The preregistered proposal-quality gate failed. The Mushroom LLM policy line stops;
no fresh 30-trial policy confirmation is authorized and no threshold is tuned.

## Frozen Endpoints

| Endpoint | Result | Frozen requirement | Pass |
| --- | ---: | ---: | :---: |
| Matched-random minus LLM h2 cost | +.039980 `[+.008967, +.077148]` | 95% lower bound > 0 | Yes |
| Strong d1-root minus LLM h2 cost | +.063384 `[-.002666, +.129476]` | 95% lower bound > 0 | No |
| Collection-root selection | 10/16 = 62.5% | >= 75% | No |
| Mean d2-opportunity recovery | .3272 `[-.0242, .6605]` | >= .60 | No |

The 32 fresh cells were balanced between uncollected non-myopic opportunities and
collected semantic continuations. The run resolved all 32 after five bounded corrected
retries. Every exactness, pairing, legality, no-reasoning, no-forced-exit, and
no-rollout-call mechanic passed. It used 37 requests, 87,842 prompt tokens, and 2,701
completion tokens at `$0` API cost.

## Mechanism

The LLM was better than matched random overall, so the semantic proposals contain
signal. They were not reliable enough to overcome a strong myopic-root policy with
exact continuation.

On the 16 uncollected opportunities, the fixed collection-root continuation was:

- `bruises`: 9 cells;
- `odor`: 6 cells;
- `spore-print-color`: 1 cell.

Only the six odor choices matched the exact collection continuation. Mean collection
follow-up regret was .2044 nats. Seven cells had negative opportunity recovery, and
the exact verifier abandoned collection in six cells.

The errors show a broader index-position heuristic. Uncollected branches used menu
index zero in 169/268 = 63.1% of choices. Collected branches selected
`spore-print-color` in 257/364 = 70.6% of choices, usually from the far end of the
menu. The strict interface therefore produced syntactically valid policies without a
stable semantic mapping from branch beliefs to informative follow-up features.

A local replay reproduces every registered selected decision, endpoint, interval, and
gate within `1e-12`. One near-solved cell swaps the unselected fourth candidate root
between exactly tied features across CPU/container arithmetic; no selected cost or
registered comparison changes.

## Scope

This does not weaken the exact Mushroom planning result: exhaustive endpoint-aligned
d2 still beats d1 on 1,000 paired rows. It shows that this particular non-thinking
26B indexed proposal interface does not transfer that opportunity into a qualified
LLM policy. The negative gate is informative evidence about the proposal bottleneck,
not a formal policy comparison.
