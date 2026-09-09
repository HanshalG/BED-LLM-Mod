# Luna medium nested-expression qualification: adapter failure

Executed once from pushed implementation f32440fc under protocol
65d045f418706c8ff115e29af60b5f8c85f5fb0ff58fe6d48559baed3bb26368.
Terminated failed_closed after six accepted OpenAI Luna medium calls, costing
$0.02757011 with zero uncertain exposure. All responses stopped normally;
reasoning tokens ranged from 1,034 to 2,954, not a thinking-limit exit.

Two initial proposal/repair/search updates completed. The third failed before
search dispatch: `Value = lbind(Value, Value, Value)` was absent from the grammar.
The model generated three-argument lbind calls, but the DSL takes two arguments.
Graph validation permits generic argument counts without checking each function
signature. Execution feedback reports such errors, but search guidance assumes
every accepted graph maps to an existing grammar production. Retaining an invalid
original program alongside repairs exposed this unchecked assumption.

This is an adapter robustness failure, not a completed semantic or planner null.
Task four was not requested; the initial coverage test was not reached. No
aware/blind refresh, forecast sealing, target labels, or depth tests opened.

Visible-example diagnostics only: task 0 had one matching initial program, three
mismatches, and an invalid repair batch. Task 1 had two initial matches, one
mismatch, one execution error, and four matching repaired programs. These are
observed-example fits, not held-out accuracy, causal repair gains, or non-myopia.
Different cohorts also prevent a controlled comparison to earlier interfaces.

Verification: 38 focused tests passed before launch. The saved failure prefix
replays exactly: six recorded calls, zero new calls. No Docker containers remain.
Raw bank: `results/nonmyopic/rearc_expression_qualification_20260909/`.
Do not mutate or rerun the bank.

Authenticated closing credits/usage/balance:
$245 / $221.418276889 / $23.581723111. London Sep9 posted spend $0.95999841;
conservative ledger spend $0.99999841 including prior $0.04 uncertainty.
Remaining allowance $4.00000159 does not authorize reopening this run.

Next: define and test total handling of ill-arity proposals before guidance,
preserving fixed proposal slots and positive grammar fallback. Use synthetic
malformed/repaired-call fixtures first. Preserve this implementation for replay;
a successor requires separately frozen semantics, not post-response filtering.
Research goal remains unachieved.
