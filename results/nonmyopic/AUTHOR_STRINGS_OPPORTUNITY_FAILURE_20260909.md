# Frozen string opportunity diagnostic: failed closed

Code was committed/pushed at 21f6fde7 before execution. Protocol SHA256
e6db22df793b2199b7a74869e6fff21673e181db616693e1a29ffd0cded317f0.
An initial direct-file invocation failed at import before run() or any artifact
creation. Correcting the invocation to python -m scripts.author_strings_opportunity
opened the run once. That process exited with the banked status failed_closed,
error_type ValueError. No code, grammar, cap or role changed during execution.

## Implementation failure

The existing program_induction proper-score helper requires nonempty categorical
keys. The new grammar permits empty-string outputs, but the adapter passed these
through as raw keys. Forecast construction and sealing completed; real-path scoring
then could raise this ValueError. A synthetic empty-string fixture reproduces it.
This was a missed integration boundary, not a model failure or a scientific null.
The runner does not retain a traceback, so the synthetic reproduction plus saved
empty forecast cells diagnoses the compatible failure mechanism, rather than claiming
an independently saved exception stack exists.

Do not replace the saved failed result with a corrected rerun. The frozen source,
prior and role allocation remain closed. A new JSON-string categorical encoder has
been added for future adapters, with injectivity, empty-string and proper-loss tests.
It is deliberately not wired into the historical runner, preserving its bindings.

## Independent support limitation

Both forecast matrices were saved before real-path scoring:

| Task | Complete syntax enumerated | Compatible syntax | Prior units | Distinct behaviors on nine inputs |
|---|---:|---:|---:|---:|
| 1 | 3660 | 282 | 636 | 7 |
| 10 | 3660 | 0 | 0 | 0 |

The declared grammar has 60 operations, all one- and two-operation sequences.
Uniform length then uniform operations is represented by 7200 unconditioned integer
prior units: each length-one syntax has multiplicity60, each length-two syntax1.
Task1's matrix contains six empty-string cells across weighted rows; task10 cannot
support reference planning at all. These counts are directly checked from the saved
forecasts, not a post-hoc support search or rerun.

Even correcting the category adapter would not make this frozen comparison satisfy
the requirement for complete references on both tasks. We have not measured a full
h1/h2/h3 result, and must not claim the task family lacks non-myopic opportunity.
The grammar's missing coverage and the scorer's input contract are separate failures.

## Next decision

Keep this as a development instrument failure and evidence that ad hoc unary-chain
string grammars are insufficient for at least one opened task. Before another source
opportunity run, use a justified established string-expression language or a separately
frozen LLM-native proposal interface and verify basic representational coverage on
development examples. Do not tune this grammar until the opened pair passes and
present that as a prospective result. Any new formulation needs fresh evaluation
tasks, the promised symbolic controls, and its own predictive/opportunity gates.

Forecast SHA256 0e567b7429a6e4f7c37e836074701ebea6541fc0bc5bd54f5fa7a6829f866ffe.
Artifacts: author_strings_opportunity_20260909/. No paid calls, no extra source tasks,
no historical runtime changes. Account and London ledger unchanged. Full LLM-native
non-myopic BED goal remains active and unachieved.
