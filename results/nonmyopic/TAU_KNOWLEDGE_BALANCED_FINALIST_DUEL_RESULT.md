# tau-Knowledge Balanced Finalist Duel Result

## Decision

The balanced finalist duel passes every efficacy and safety criterion but fails
both preregistered decision-coverage criteria. The overall development gate
therefore **fails**, and no fresh task is opened.

The frozen policy is strongly directional:

- selected-root oracle-tail total: `38` versus `31` myopic (`+7`);
- task wins / ties / losses: `3 / 17 / 0`;
- selected-root-versus-all accuracy: `.8077` versus `.6214`;
- accuracy gain: `+.1863`; and
- all three nonzero task effects favor the duel policy (`+2`, `+2`, `+3`).

However, only `6/13` changed-finalist tasks produced the same canonical winner
in both A/B orders. The gate required at least `8/13`. The remaining `7/13`
used the safe myopic fallback, above the allowed maximum of five.

## Frozen Gates

| Criterion | Result | Status |
| --- | ---: | --- |
| Unanimous canonical duels | `6/13` | fail (`>=8`) |
| Position/tie fallbacks | `7/13` | fail (`<=5`) |
| Oracle-tail gain over myopic | `+7` | pass (`>=4`) |
| Wins minus losses | `+3` | pass (`>=3`) |
| Losses | `0` | pass (`<=1`) |
| Selected-versus-all accuracy | `.8077` | pass (`>=.70`) |
| Accuracy gain over myopic | `+.1863` | pass (`>=.08`) |

The exact one-sided sign-flip value over the three nonzero task effects is
`p=.125`. A 100,000-draw task bootstrap gives a total-gain interval `[0, 15]`.
These are descriptive because the development endpoints were already open.

## Position-Bias Diagnosis

The policy behaved as designed: a disagreement or tie could never override
myopic. Both serving tasks exhibited position instability, and development
retained seven fallbacks. The six position-consistent decisions comprised four
non-myopic overrides and two myopic confirmations. The four overrides yielded
three gains and one tie; none lost.

Thus the semantic comparator can identify high-value future-visible roots, but
side-by-side candidate position remains too load-bearing to satisfy the frozen
coverage requirement. Relaxing the coverage threshold after observing the
endpoint would be invalid.

## Integrity And Cost

Serving:

- exact logical requests / HTTP attempts / retries: `4 / 4 / 0`
- prompt / completion / reasoning tokens: `17,116 / 390 / 0`
- forced exits: `0`
- cost: `$0.04864`
- private raw SHA-256:
  `282674e5e3b17f3d86a0bc96f450f06af2964f1292d017a2d0466dff7ea25007`

Development:

- exact logical requests / HTTP attempts / retries: `26 / 26 / 0`
- prompt / completion / reasoning tokens: `111,483 / 2,462 / 0`
- forced exits: `0`
- cost: `$0.2810775`
- private raw SHA-256:
  `d9597d38e05df1071da02c257e7d142cf1655bfa1971decfb0ec71dd81714780`

Combined spend: `$0.3297175`. Authenticated post-run OpenRouter balance:
`$29.668149594`. There is no fixed reserve.

## Next Method

Close the exact side-by-side balanced duel. A distinct successor may remove
candidate position entirely by scoring each finalist independently against the
same pooled information-need support:

- label every shared need as covered, partial, or unsupported from one
  candidate's retrieved evidence;
- repeat each candidate map independently;
- compare fixed discrete support totals; and
- override myopic only when both non-myopic totals exceed both myopic totals.

That tests a shared semantic belief/value scale rather than an A/B preference.
It requires a separate preregistration and remains development on open tasks.
