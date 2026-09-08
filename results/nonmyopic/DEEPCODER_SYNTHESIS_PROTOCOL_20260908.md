# CrossBeam history-guided search: exploratory comparison

Freeze this implementation before search on the64 already-banked rejection
histories. This is exploratory reuse of opened histories, not fresh confirmation
or a repair of earlier gates. Verify the rejection artifact SHA256
a4d8b2e575762b22b1593c71a6ea95a3a5c1406fb2e2209c7b95a3de1ef1c9c9
and each reconstructed history hash before searching. Do not rerun rejection.

Reuse CrossBeam's Apache-2.0 baseline enumerator and Value implementation at
c43cb523fa9887513fb18bf088eb588f3fec5835. Load only these two hash-verified source
files, no datasets or neural model. Interpreter remains pinned ExeDec DeepCoder.

Bind all published DeepCoder operations and compatible fixed lambdas. Each
partially evaluated value retains its static type even if its first example
returns ERROR. ERROR propagates through expression arguments. Native operation
exceptions propagate; they are not silently converted into mismatches. Result
trees are interpreted directly, never evaluated as generated Python.

Budget:2048 operation applications, expression-tree weight<=9, five seconds per
search. Each input reference and operation has weight1; a lambda is part of the
operation. Enforce the application cap at every operation, including within the
upstream Cartesian loop. Also count individual-example evaluations and time.
This is not compute-matched to2048 prior program draws; compare finding rates
descriptively and report actual work, not equal-budget efficacy.

CrossBeam merges expressions that agree on observed examples and returns the
first matching expression. This is ONE candidate, not16 posterior samples. It
may have a different syntactic representation from the original2--4-statement
prior, and is not a conditional-prior sampler. Explicitly permit input-identity
solutions and full-vocabulary trees within the search cap. No outcome-specific
grammar changes or filtering of errors/contexts.

Report any-candidate finding rates versus whether rejection found at least one
match, not versus whether it filled16. On found expressions only, fix target
predictions before evaluating targets and report point-predictor half-Brier
(0/1 exact output mismatch). This selected-subset error is not a learning curve,
posterior calibration or an all-case policy score. Store found expressions and
prediction hashes; exceptions bank failed_closed and no retry is allowed.

Reserve output first, checkpoint all64 contexts. Old histories may already have
opened target outcomes; this freeze prevents new algorithm tuning during this
comparison but does not make it prospective independent validation. No pass
threshold or paid authorization. No cluster, external deployment or automation.
