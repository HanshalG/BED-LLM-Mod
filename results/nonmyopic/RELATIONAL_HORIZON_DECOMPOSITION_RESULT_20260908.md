# Exact diagnosis: first-query rank reversal and insufficient total headroom

Retrospective protocol/code frozen3fbe5b51. Result SHA256
dbadd7a95722de2a9b4a034312c9a823adadb7708fef5fec66f778909acdb8dd.
All four original likelihood matrices were reconstructed and matched their
banked hashes before reference computation. No new seeds, labels, task selection,
LLM calls or efficacy rerun. The parent remains a frozen opportunity null.

## Where h3 Loses To h2

Panel0 has a genuine reversal of first-query ranking when moving from the
three-measurement truncation to the actual four-measurement budget:

| First query | Optimal risk after 3 measurements | Optimal risk after 4 measurements |
| --- | ---: | ---: |
| 1, selected by h2 | .05505979 | .03657069 |
| 3, selected by h3 | .05396997 | .03687652 |

h3 correctly optimizes its three-step objective. That objective prefers query3,
but query1 has better four-step value. h2 picks query1. Both deployed policies
then attain the optimal continuation conditional on their root. The entire h3
loss of.00030582838 is therefore root regret, not poor subsequent replanning or
noisy inference. Root IDs are zero-based as in the banked action menu.

This demonstrates the distinction between exact optimization of a truncated
objective and optimal deployment over a longer budget. Longer truncations do
not necessarily preserve root rankings under the actual terminal objective.

## Complete Decomposition

For each policy, achieved risk minus exact four-step optimal risk equals root
regret plus continuation excess. All identities hold as exact fractions.

| Panel | Optimal B4 | h2 root regret | h2 continuation excess | h3 root regret | h3 continuation excess |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | .03657069 | 0 | 0 | .0003058284 | 0 |
| 1 | .02867455 | 0 | 0 | 0 | 0 |
| 2 | .03915718 | .0012157147 | 0 | .0003500663 | 0 |
| 3 | .04704096 | .0000649171 | .0000434036 | .0000878142 | 0 |

After h3's first measurement only three measurements remain, so h3 must have
optimal continuation in this exact model. The zero continuation excess on all
four panels verifies this invariant. On panel3, h3 chooses a slightly WORSE root
than h2 but compensates by removing h2's continuation error. A single selected-EIG
or aggregate endpoint metric would conceal those distinct effects.

## Stronger Stopping Evidence

The aggregate full-budget optimum is.0378608450, compared with myopic.0394558121.
Even an exact four-step planner can improve on h1 by only4.0424% on this banked
finite population. From h2, only0.8667% remains; from h3, only0.4887% remains.

Thus neither perfect h3 estimation nor more rollout computation can deliver the
frozen5% first adjacent gain on these SAME tasks and this SAME prior/objective.
There is insufficient total headroom, not just a poor h3 approximation. This is
a finite-panel exact bound, not a statement about the entire grammar distribution
or an LLM's misspecified deployed beliefs. No paid model sweep follows.

This resolves the immediate diagnostic question. Stop further same-panel depth,
seed, estimator, grammar-weight and root-choice variants. In a future genuinely
new formulation, check full-budget-vs-myopic headroom before a depth ladder when
an affordable exact reference exists. For an expensive reference, require a
justified headroom estimate/bound, not a syntactic complexity count. This changes
the order of future dependency checks, not any historical gate or result.

Do not fix the desired curve by renaming recursive full-budget policy improvement
as horizon depth. Approximate full-budget rollout with a baseline terminal value
may be a useful architecture, but it is a distinct treatment requiring a new
prospective claim and fair baseline, not a repair of these failed h1/h2/h3 gates.
The original LLM-native non-myopic goal remains unmet.

## Verification

10 focused tests passed in0.28s, scoped lint passed. Tests check decomposition
identity/nonnegativity, exact h3 continuation, root tampering rejection and the
prior adapter's independent-planner comparisons. A test fixture initially asked
the without-replacement random control for more queries than its menu; it was
corrected before freeze to the four-query/four-budget supported setting. No
scientific setting changed. Reference computations totaled0.082s across panels.

Account refreshed245/220.376693994/24.623306006; London Sept8 spend0, remaining$5.
Model calls0; process exited successfully. Parent hashes/status unchanged, no
cluster use, automation paused, full goal unfinished.
