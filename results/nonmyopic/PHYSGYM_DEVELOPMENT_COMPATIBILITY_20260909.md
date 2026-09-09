# Four-task source compatibility and safe proposal evaluation

Previous turn established the wrapper contract. This turn opened a prospectively
selected four-task development sample and implemented a bounded scalar-expression
interpreter. It did not run a model or test experimental-design efficacy.

## Source selection

Selection scope was written before downloading full_samples.json. IDs sorted by
SHA256('physgym-dev-v1:'+id) select103,458,457,653 from97unique source IDs. The full
JSON was parsed locally but only selected records were displayed; neither solutions
nor answers were emitted. Remaining task content was not inspected or used in
selection. This is not a sealed ingestion claim. Manifest records source SHA256
7903440c59e4379f414b88704a01e9ba7f56adca7a5d6d363a614bd2f92afa12 and pinned commit
fe68079c0921029dde679ed44bc3192dd3b270ab. Root LICENSE was read and declares MIT;
separate original PHYBench data provenance remains an adoption consideration.

| Task | Source form | Inputs | Consequence |
|---|---|---:|---|
| 103 | Constant times thermal resistance | 1 | Strong linear symbolic baseline; likely little measurement opportunity |
| 458 | Monomial triangular-pipe flow | 4 plus dummy density | Strong log-linear baseline; physics regime must be respected |
| 457 | Trigonometric polygon collision expression | 4, including integer N | Nontrivial formula, but difficulty alone is not non-myopic structure |
| 653 | Square root of gravitational plus magnetic terms | 6 plus dummy velocity | Small-ring/strong-field assumptions matter for valid experiments |

Keep all four, including the easy linear task. No difficulty-based replacement.
No finite sampling box is provided by these source records. Code assertions give
positivity and integer-N restrictions, not the full physical regime. Rich public
context specifies the governing setup and may enable textbook recovery; any later
result must distinguish semantic recall from observation-driven proposal improvement.

## Numerical instrument

ScalarExpression uses Python's AST parser plus a bounded arithmetic interpreter,
not eval/exec, generated imports, arbitrary attributes or an upstream sandbox.
It allows declared scalar names, numeric constants, pi, arithmetic and six math
functions; optional np spelling is normalized only for those allowlisted symbols.
Limits:8192bytes,512ASTnodes,depth32,16variables,absolute evaluated exponent<=32.
Arithmetic operates on floating-point values, rejecting nonfinite inputs/results,
booleans, invalid domains, arbitrary calls and explosive exponents. It is not a
general Python sandbox and intentionally cannot represent arbitrary source code.
Existing ChemBench IR remains unchanged; its fixed inputs/functions did not cover
the trigonometric source and changing it would alter unrelated historical machinery.

On each selected task, interpreted stored equation and extracted single return AST
agree exactly on32 fixed diagnostic points. All outputs are finite. No benchmark
Python function executes, though the source mathematical expression is deliberately
interpreted. This checks expression/return compatibility, not correctness of the
underlying physics, source guard enforcement or accuracy outside these points.
The diagnostic box(.5,2), integerN3..12, is NOT a physically validated experiment
distribution; in particular it need not satisfy the small-ring regime. The JSON
explicitly records scientifically_valid_input_distribution=false.

Fifteen tests pass(.15s) for metadata-only selection, numeric semantics, unsafe AST
rejection, invalid arithmetic, resource bounds and input validation; lint passes.

## Next dependency

Before paid context-versus-blind proposals, prospectively specify and justify
admissible distributions for all four tasks, including dummy variables, with
explicit source assertions and asymptotic-regime limitations. Alternatively frame
the task strictly as source-function induction on code-valid inputs, never as
validated physics; that scope choice must be made before responses, not after.

Then freeze a small paired semantic/history intervention: identical observations
and proposal budgets, full-context versus anonymized/no-context generation, plus
no-new-observation matched-width sampling and a simple symbolic predictor. Keep
independent target inputs and a proper prediction endpoint. Raw context superiority
alone is not enough: require observations to cause useful hypothesis changes beyond
the same-budget controls. No reference planner is needed until this first link
passes. No paid calls are authorized merely by the compatibility audit.

Account245/221.179955339/23.820044661 unchanged; conservative London-day remaining
4.23832314, old.04uncertainty retained. Calls0/cost0. Cluster, automation and protected
historical runtime untouched. Goal remains active/unachieved; this turn produced
new source evidence and a tested interpreter rather than another status restatement.
