# CrossBeam executable-search comparison

The exploratory comparison frozen at `cc681060` completed all64 banked histories.
Every history hash matched the preceding rejection artifact. This is new search
on opened contexts, not independent confirmation or a rescue of an earlier gate.

| Observations | CrossBeam found a fit | Rejection found any fit | Search found-only point Brier |
|---|---:|---:|---:|
| 1 | 13/16 | 13/16 | .57212 |
| 2 | 11/16 | 12/16 | .47443 |
| 3 | 7/16 | 8/16 | .20982 |
| 4 | 5/16 | 7/16 | .15625 |

The rejection comparator is ANY accepted program, not filling16 particles.
The search errors describe changing found-only subsets and must not be called
a learning curve or compared directly against rejection's different completed
subsets. Both fit the same observed histories; neither receives held-out targets
while searching. No fits were discarded based on predictive quality.

Search used66672 operation applications and194397 example evaluations in .533
seconds summed across64 contexts. No call exceeded2048 applications. Rejection
used107117 complete program draws in9.376 seconds. These different work units
are NOT a matched-compute efficacy comparison. The runtime advantage describes
these capped implementations and tasks only, not general synthesis scalability.

## Implemented baseline

`environments/program_induction/synthesis.py` reuses the hash-pinned CrossBeam
enumerator, not a new custom search algorithm. It binds all DeepCoder operations
and compatible lambdas to ExeDec's bounded interpreter, retains static types
through ERROR observations, and independently replays each found expression
against the full observed history. New inputs are evaluated directly on the
expression tree; generated Python is never executed.

The upstream work check is outside its innermost Cartesian loop. The adapter
adds per-operation checks and propagates errors rather than letting a large
partition silently exceed the requested cap. Imports fetch two source files only;
they do not install CrossBeam's model stack, datasets or checkpoints.

CrossBeam merges values that agree on observed examples and returns the first
fit. This is a useful program-search baseline, but NOT a posterior sampler or
diverse predictive hypothesis pool. Its expression-tree search budget also differs
from the2--4-statement data-generating prior. Those limits remain explicit.

## Consequence

The next inference method must maintain alternative executable explanations with
different predictions beyond the observed examples. Finding one convenient fit
does not repair the support problem; assigning it probability1 would manufacture
certainty. Productive search is now available as a baseline, but these results
do not support proceeding to another depth sweep or paid LLM study.

Do not rerun these contexts with a larger search weight/cap or treat the fastest
found-only subset as a new positive endpoint. A separate prospective predictive
pool contract is needed, including prior/selection interpretation, independent
held-out calibration and computation accounting. Unknown search or LLM proposal
probabilities cannot be described as exact Bayesian posterior weights.

## Verification and provenance

18 focused synthesis/rejection/interpreter tests passed in .69 seconds; lint
passed. Tests include new-input replay, ERROR preservation, strict operation
caps, input-identity solutions and malformed history rejection.

CrossBeam commit c43cb523fa9887513fb18bf088eb588f3fec5835, Apache-2.0:
- value.py SHA064bf03692832ea814bf6a5301553f92143f730b218a2a733359ee37e555a1ee.
- baseline_enumeration.py SHA9f81d63a56ba98f7599152663b06ee16533eb9019d03ba36c94944c4380b5a61.

Artifact `DEEPCODER_SYNTHESIS_AUDIT_20260908.json` SHA256
`3ab65510c754d2be801c90bb135b114ecf148f885c76ec622c7cbfe651b031d1`.
It stores all histories' statuses/work and found expressions/prediction hashes.
The process exited normally. No model calls, $0 spend; authenticated account
245credits/220.376693994usage remains consistent with the Sept8 ledger.
No cluster, protected runtime or automation change. The full goal is unfinished.
