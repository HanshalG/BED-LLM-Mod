# Contradiction recovery: fixed-length neighborhoods are insufficient

Retrospective mechanics on all128 already-opened fresh-screen outputs. Previous
expanded pools remain immutable. New repair prepares a one-hop neighborhood of
those pools using only the two previous observations, then filters on each third
observation independently. This preserves contradicted roots as search seeds,
unlike filtering the roots first. Candidate generation has no new-answer input.

| Case | Previous support | Repair candidate support | Unsupported before | Unsupported after |
|---|---:|---:|---:|---:|
| 0 | 18 | 76 | 0 | 0 |
| 1 | 5 | 8 | 14 | 14 |
| 2 | 36 | 125 | 0 | 0 |
| 3 | 4 | 6 | 3 | 3 |

No unsupported observation is recovered. This is not a prospective predictive
score or evidence that repair can never work. A focused synthetic test shows
preserving an incompatible root can recover a compatible neighbor where the
original filter-first expansion returns no pool. It just does not repair these
empirical missing models. No extra mutation depth or selective cases tried.

Post-result inspection of source programs (only already-opened screen cases):
case1 truth uses Scanl1(-), Map(square), ZipWith(-), ZipWith(+) in four steps;
Luna proposes Filter(even), Map(square), ZipWith(-) in three. Case3 truth uses
ZipWith(*), Scanl1(min), Sum; Luna proposes ZipWith(*), Sum. The fixed-length
neighborhood cannot contain either exact true syntax, regardless of how many
same-length substitutions are enumerated. This is not proof no shorter program
is behaviorally equivalent, but the measured coverage deficit remains.

Next justified intervention is a source-valid structural proposal interface
(insertion/deletion or full regeneration) conditioned on newly observed residuals,
with a fresh prospective predictive/transition test. Do not alter the completed
predictive endpoint to claim a rescue. True source programs in this retrospective
analysis must never enter policy prompts or repair inputs. Non-myopic discovery
requires validating these transitions before counting their anticipated benefit.

One synthetic regression passed in0.64s; scoped lint passed. API calls/cost0,
no new hidden target answers. Authenticated usage220.432893399; conservative
day spend$.091814205 still exceeds posted$.056199405 and includes the prior$.04
uncertainty. Previous goal turn progress; this turn adds a tested mechanism and
empirical structural diagnosis. Goal incomplete, no headline or depth claim.
