# Missing behavior dominates the feedback screen's error

Retrospective diagnostic of all eight already-opened feedback cases and all
three arms. No new outcomes, model calls, deployed weights or reruns. The previous
turn completed the paired feedback null; this turn measures whether probability
weighting alone could plausibly fix it.

## Optimistic ceiling

For each case allow an oracle to choose ANY mixture over the retained executable
program behaviors, using all32 true target answers. Use one shared vector of
weights across targets, not independent per-target program selection. Minimize
the same half-Brier endpoint loss. This deliberately gives reweighting an
unrealistically favorable diagnostic, not a valid forecasting procedure.

| Arm | Saved mean Brier | Oracle lower bound | Feasible oracle upper bound |
|---|---:|---:|---:|
| Initial | 0.208059321 | 0.183227536 | 0.183227539 |
| Feedback | 0.208094152 | 0.183227536 | 0.183227539 |
| Control | 0.208146457 | 0.183227536 | 0.183227539 |

Roughly88% of current error remains even with hindsight-optimal weights on
these supports. An analytic per-target support relaxation alone gives a mean
floor0.1826171875. The shared-weight constraint slightly strengthens this floor.
The largest reweighting opportunity is case4, not the main missing-support cases.

| Case | Feedback saved loss | Best mixture loss (approximately) | Feedback distinct behaviors |
|---|---:|---:|---:|
| 0 | 0.000559 | 0 | 4 |
| 1 | 0.437500 | 0.437500 | 1 |
| 2 | 0 | 0 | 1 |
| 3 | 0 | 0 | 1 |
| 4 | 0.186378 | 0 | 12 |
| 5 | 1 | 1 (fixed abstention) | 0 |
| 6 | 0.040316 | 0.028320 | 15 |
| 7 | 0 | 0 | 1 |

All three arms have the same best mixture value. More syntax produced by the
revisions did not improve attainable predictive performance on this finite
target panel. This is not proof of equivalence on all possible inputs or an
intrinsic lower bound for the whole program grammar. It is a bound restricted
to saved support and the existing abstention convention. New behaviors, a new
observation, or a broader forecasting model can evade it.

## Concrete structural failure

On case1 all three initial observations happened to make the true two-statement
program `Last(x0); Drop(last,x0)` return x0 unchanged: all observed final elements
were negative. The compatible retained pool instead contains identity behaviors
such as double reversal. Every retained program agrees on the target panel,
yet that behavior is wrong on14/32 targets. No weighting can fix a one-behavior
pool. The source truth is already opened; it was NOT used to build the supports.

The local substitution expansion preserves intermediate output types. Moving
from double reversal to Last/Drop changes the first result from a list to an
integer and changes its consumer. A coordinated typed rewrite is outside this
one-statement neighborhood. This is a concrete representational limitation,
not missing floating-point precision or insufficient reweighting iterations.

Case5 remains an empty support. Its opened source program computes
`Map(*2,x0); ZipWith(+,x1,doubled); ZipWith(min,doubled,sum)`. Both revision arms
failed to produce any program fitting all three public examples. Case4, in
contrast, admits a behavior matching all32 targets after local expansion, so
its error can in principle be reduced by reweighting. These mechanisms differ.

## Numerical method and verification

Write Q_ij as the fraction of targets on which programs i,j agree, and b_i as
the fraction where program i equals the target truth. The convex objective is
`f(w)=(1+w'Qw-2b'w)/2` on the simplex. SciPy1.14.1 SLSQP supplies a feasible
candidate. Its gradient g gives a convex lower bound `f(w)-(w'g-min(g))`.
Independently, when the correct category is absent and k distinct wrong outputs
are available at a target, every mixture has loss at least `(1+1/k)/2` there.
The reported lower bound combines these checks. Syntax-equivalent behaviors are
deduplicated only inside this hindsight optimization, not in original priors.

Floating-point bounds have a small numerical allowance; these are not formal
interval-arithmetic proofs. The mean interval width is3.12e-9. Per-case optimizer
flags and gaps are retained; no success flag is accepted in lieu of a gap. The
case6 width is about2.5e-8, so its strict1e-8 resolution flag is false. This does
not affect the much larger substantive conclusion and no retry was run.

Five tests passed1.23s: analytic missing-category floor, shared-weight versus
independent-target distinction, correct-world optimum, duplicate/permutation
invariance, empty support and shape rejection. Scoped lint passed. The audit
first replayed the complete paid parent b85bd72c, then reconstructed each
support from raw responses and source execution. Every bound was checked
against the feasible saved forecast. All processes exited.

## Next decision

Do not spend on a weight-only rescue or another undirected feedback/breadth
increment. The next constructive mechanism should explicitly seek behaviorally
different, history-compatible executable hypotheses on public unlabeled inputs,
including coordinated intermediate-type changes, with a genuinely competitive
symbolic search control. Its ability to create new behavior must be measured
before a new paid joint simulator test. This is an architectural direction, not
an already-qualified new interface or a claim that the LLM will beat that control.
The original goal still requires reliable joint predictions, useful query
ranking, and an actual controlled non-myopic result; this oracle is none of those.

Cost0; authenticated account245/220.663020549/24.336979451 unchanged. LondonSept9
recorded0.24474207 includes prior uncertainty; remaining4.75525793. No cluster
or automation changes. Goal active/incomplete.
