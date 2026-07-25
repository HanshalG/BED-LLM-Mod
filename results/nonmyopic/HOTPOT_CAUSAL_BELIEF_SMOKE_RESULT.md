# HotpotQA Causal Belief-State Smoke Result

Date: 2026-07-25

## Outcome

**The exact ten-call smoke passed every mechanics and causal-state check but
failed the conjunctive scientific gate.** The 100-task development split and
5,318-task holdout remain sealed, and this exact interface is closed.

Public artifact:

`results/nonmyopic/hotpot_causal_belief_smoke/hotpot-causal-belief-smoke-20260725T041356Z/SMOKE.json`

## Mechanics

| Metric | Result |
|---|---:|
| Physical requests / HTTP attempts | 10 / 10 |
| Transport retries | 0 |
| Reasoning tokens / forced exits | 0 / 0 |
| Parsed responses | 10 / 10 |
| Distinct refreshed states | 4 / 4 |
| Aligned vs shuffled score vectors differ | 4 / 4 roots |
| Aligned vs initial score vectors differ | 4 / 4 roots |
| Prompt / completion tokens | 11,185 / 3,422 |
| Cost | $0.0792925 |

The branch intervention was real: all four paragraph-conditioned hypothesis
sets differed from the initial state and from one another. The state-only
continuation scorer changed its vector under both aligned-vs-shuffled and
aligned-vs-initial interventions on every root.

## Frozen Policy Result

The answer-bearing document was root 0 and the enabling document was root 1.
The aligned continuation after root 1 correctly selected root 0, satisfying
the intended enabling-to-answer link. However, the converse semantic shortcut
also appeared: after answer root 0, the aligned scorer selected enabling root
1 even though the frozen exact title-link transition could not.

| Policy | Root | Follow-up | Support coverage |
|---|---|---|---:|
| Myopic receding | answer | enabling | 2 |
| Fixed-support d2 | answer | enabling | 2 |
| Model-aware d2 | answer | enabling | 2 |
| Shuffled-belief d2 | answer | enabling | 2 |
| Random receding | enabling | answer | 2 |

The root score arithmetic explains the tie in behavior:

| Root role | Immediate | Best aligned continuation | Model-aware total |
|---|---:|---:|---:|
| Answer | 98 | 56 | 154 |
| Enabling | 42 | 100 | 142 |
| Distractor 1 | 0 | 100 | 100 |
| Distractor 2 | 1 | 99 | 100 |

Model-aware d2 therefore retained the myopic answer root. It achieved exact
coverage 2, but did not select the enabling root or exceed myopic coverage.
The final answer, "the Army and the Navy," had token F1 `.3333` against the
hidden answer and was diagnostic only.

## Interpretation

The smoke establishes a clean causal second link: observations change
natural-language beliefs, and those beliefs change continuation scores without
the scorer seeing the question or paragraph. It does **not** establish a
non-myopic first-action advantage. The LLM's semantic inference is stronger
than the audit's sole-title-link transition and lets a receding myopic policy
recover from the answer-first root.

Hard-restricting legal follow-ups to explicit title links would manufacture the
desired gap but make the LLM unnecessary, contrary to the project objective.
Changing score weights or selecting another public task after seeing this
failure would tune the endpoint. Neither is pursued. This exact Hotpot causal
bottleneck is closed before development or holdout spend.
