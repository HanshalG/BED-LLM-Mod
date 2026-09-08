# Independent Luna joint simulator: complete screen, gates not passed

Executor frozen at 4f025036, London-day ledger opened at 10ac0531 before calls.
Protocol SHA513b6d0efc84c942835cbf5b62a463e6f6e1fa8a07cb4ee46fa1e70d028ae9cf
was unchanged throughout. All eight fresh cases and32 medium-reasoning Luna
calls completed without retry, fallback, schema or transport failure. Endpoint
forecasts were sealed before target outcomes. This is a complete diagnostic null,
not an interrupted comparison and not a non-myopic policy test.

## Joint teacher results

| Teacher | Supported actual query answers | Mean conditional target half-Brier | Zero-mass targets /256 |
|---|---:|---:|---:|
| One locally expanded pool A | 5/8 | 0.600044 | 157 |
| A plus structural insertion | 5/8 | 0.596348 | 147 |
| Independent pools A+B | 7/8 | 0.397627 | 93 |

Adding the second independently seeded proposal call improved descriptive mean
loss by33.73% relative to A and rescued answer support in cases0 and6. All arms
retain unsupported-answer penalties of1; no supported-only selection. A+B has
more model compute than A/insertion, so this is evidence for proposal breadth,
not an equal-compute planning advantage or a full-grammar Bayesian posterior.

| Case | A+B actual-answer probability | A+B conditional Brier | Regeneration actual Brier | Repeat actual Brier |
|---|---:|---:|---:|---:|
| 0 | 0.995575 | 0.677083 | 1.000000 | 0.677083 |
| 1 | 0.767802 | 0.250844 | 0.272606 | 0.265625 |
| 2 | 0.600000 | 0.013889 | 0.014752 | 0.013889 |
| 3 | 0.999950 | 0.071364 | 0.133688 | 0.202462 |
| 4 | 0.958333 | 0.643667 | 0.687500 | 0.687500 |
| 5 | 0.997105 | 0.522194 | 0.508834 | 0.517361 |
| 6 | 0.685185 | 0.001975 | 1.000000 | 1.000000 |
| 7 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |

The joint prediction gate failed: answer coverage7/8 not8/8, target log loss
infinite because93 target outcomes lack mass, and mean Brier0.397627 exceeds0.15.
The nonworse-versus-controls criterion passed but cannot rescue the conjunction.
High actual-answer probability is not enough: cases0/4/5 still forecast targets
poorly after that answer.

## Regeneration and ranking

Actual mean Brier: filtering0.600044, insertion0.596348, regeneration0.577172,
repeat-old-history0.545490. Regeneration is worse than the repeat control in
aggregate on this fresh panel. Of the pairs with absolute difference>0.01,
case0 favors repeat, case3 favors regeneration. A+B predicts both directions
correctly, but only two informative pairs exist, below the required three.
The updater-ranking gate is therefore insufficient, not passed. It is not a
query-ranking measurement. Cases1/2/5 have subthreshold differences;4/6/7 tie.

The earlier one-case fresh-history discovery remains real evidence from its
own panel, but this screen does not replicate an aggregate regeneration benefit.
Equal call/token caps do not guarantee equal consumed reasoning tokens.

## Decision

Do not launch a depth grid from this teacher and do not rerun this interface
with more seeds/calls or softened criteria. The extra independent pool helps,
but neither breadth nor structural expansion has produced reliable conditional
world predictions. The next genuinely changed interface should address proposal
quality, for example executable public-history feedback during construction,
with a newly frozen control-matched test. That is a candidate direction, not a
new paid runner or proof it will work. Preserve the full non-myopic goal and
its equal-budget, compute-matched, paired requirements; do not substitute this
proposal improvement as the requested result.

## Replay, tests and budget

Independent replay verified all32 request bodies/history assignments/seeds,
source compilation and expansion, both pre-answer teacher construction and
conditional forecasts, source query/target identity, forecast seals, every
score/gate, and total accepted cost. Terminal SHA:
0c361b625830d76644739fd73064a52639b85d8c9cbed93a7e34199fb7687a27.
Forecast SHA638f70fd69c769721c33c08a6b2a2e85007a6ef31254eb33ba5c8db8e453854b.
The verifier refuses incomplete/failed runs before any source/endpoint replay.

Prelaunch34tests passed in9.31s; verifier access guards2tests in0.32s; scoped
lint passed. Both paid process and replay process exited normally.

Actual cost0.13893722, no new uncertain request. Authenticated credits/usage/
balance245 /220.597215699 /24.402784301. LondonSeptember9 posted spend0.13893722,
conservative recorded0.17893722 includes the prior0.04 uncertainty, remaining
4.82106278. No borrowing or inferred top-up; no cluster/automation launch.
The exact interface is closed. The full research goal remains active/incomplete.
