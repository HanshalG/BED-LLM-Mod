# Guided multi-hypothesis collection: integration passes, control not qualified

Implemented search guidance using Herb's existing MLFSIterator, not a new
enumeration engine. All353rules have positive weights. The shared source-derived
base gives weight64 to I,16 to direct Grid-returning calls,4 to other calls,1
to other terminals. Guidance mixes50% normalized base with50% operation/terminal
counts from validated proposals. This is a search heuristic, NEVER the numerical
Bayesian prior. The guide for this test is explicitly handcrafted:
hconcat(vmirror(I),I). No model was asked to discover it.

The initial64-candidate runs spent most slots on bare function/constant roots.
Those outputs are retained. Both arms were then given the same root-only mask:
exclude bare terminals other thanI. All constants/functions remain available as
intermediates, all direct and computed calls remain allowed. In this pinned DSL
the excluded constants are scalars or coordinate pairs, not grids. This is an
engineering correction, not an outcome-dependent scientific rerun or a restriction
to a smaller set of benchmark tasks.

## Actual integrated check

Each root-masked arm emits64candidate expressions, max_depth5/max_size12. Nested
expressions convert to strict executable graphs without executing candidate code;
computed calls and repeated-subexpression sharing are supported. A bounded
collector counts invalid/duplicate slots, deduplicates canonical programs, and
uses the existing exact public-history conditioning with a uniform unique-program
prior. It never stops at the first matching program or resets a failed posterior.

Handcrafted public observation:[[0,0]] maps to[[0,0,0,0]]. Future input[[1,2]]
is used only to inspect disagreement, not compare against a hidden truth.

| Arm | Candidate attempts | Consistent distinct programs | Distinct future outputs |
|---|---:|---:|---:|
| Source-weighted control |64|0|0|
| Handcrafted guide |64|10|4|

All10guided survivors receive weight.1. There were137distinct graph/input sandbox
executions including future predictions, with exact shared caching across arms.
Candidate slots are matched, not proven equal internal search work or wall time.
No benchmark examples, LLM calls or new spend. This is a mechanism positive
control, not evidence of LLM efficacy, semantic generalization or non-myopia.

## Important unresolved baseline issue

Recorded product-rule log weights increase between successive emitted programs
at2positions in the control and13in the guided arm. Therefore the upstream
iterator's output is NOT globally probability-ordered under these saved weights,
despite its description. A source inspection shows uniform sub-iterators are
requeued using the priority of the emitted program; this is a candidate explanation,
not a proven complete diagnosis of all ordering errors. Fixing only that line
may be insufficient because within-shape enumeration also matters.

Do not interpret the table as beating a qualified classical search control.
Before a new paid proposal study, either correct and independently validate the
existing backend's ordering or use a verified alternative, retaining the same
full-language and candidate-accounting contract. Check globally ordered output
against exhaustive tiny grammars, especially equal-shape product choices, and
count internal work. Do not choose a weak ordering to manufacture an LLM win.

The completed nested-expression interface also removes the old need for a model
to invent consistent temporary variable IDs. It has strict name/call/size checks;
attributes, lambdas, lists, imports, keywords and malformed computed calls fail.
It has NOT yet been used to repair or rescore closed paid cohorts.

21focused tests pass in1.98s, including multiple survivors, duplicate accounting,
exact candidate cap, empty-support failure, nested/computed calls, and positive
fallback weights. Real Julia runs used the pinned image/dependency manifest,
no-network/read-only/nonroot isolation,2GiB/2CPU and90second timeouts. Hodel
execution used its existing per-program sandbox. All containers exited.

Both previous/current goal turns made progress. Balance23.609293221 and daily
remaining4.0275717 unchanged. Full goal active/unachieved. No cluster/automation
changes. Next dependency is the ordering/control correction, then a genuinely
new LLM proposal qualification with paired controls before depth experiments.
