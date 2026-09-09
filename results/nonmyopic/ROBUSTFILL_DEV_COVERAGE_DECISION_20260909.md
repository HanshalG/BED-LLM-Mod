# Established string language: retrospective coverage witnesses

The previous goal turn only restated banked Luna results (no progress). This turn
completed a new zero-call representation audit on the two already opened author
string tasks, not a rerun of the failed finite-grammar opportunity comparison.

Source: Google DeepMind ExeDec RobustFill DSL at commit
ef046ce2cc3fcd024e32f5dfe00e69700dac82ed, tasks/robust_fill/dsl.py, SHA256
58fc2606ba386f8b8b40acf0e874d2813cd3f3d8d07d803aa0e6711addd9a4ce.
The entire source was inspected before hash-verified execution. Its header declares
Apache-2.0. No model-generated Python or Prolog executed; no new task was opened.

## Measurement

Both tasks used all ten input/output pairs for search. This is hindsight
representability, not generalization, a belief prior, or a predictive test.
The declared source-language subset includes GetAll, GetToken, Trim, character
constants, SubStr over source positions -100..100, optional ToCase composition,
and concatenations of at most six pieces. It excludes other RobustFill constructs.
Search merges atom behaviors across the ten examples and searches shared output
prefix positions. The 10,000-state cap reports incomplete rather than absence.

| Task | Atom expressions tried | Compatible piece behaviors | Witness | Exact examples |
|---|---:|---:|---|---:|
| 1 | 162043 | 49 | GetAll(WORD) | 10/10 |
| 10 | 162043 | 303 | GetToken(WORD,-3) + space + GetToken(WORD,-1) | 10/10 |

The search reached 1/658 states respectively and independently executed each
concatenated witness against every pair. Raw outcomes and bindings are in
ROBUSTFILL_DEV_COVERAGE_20260909.json. Three synthetic tests pass, including shared
program consistency, concatenation, empty pieces, and cap handling; lint passes.

## Decision

The old missing support was a language limitation, not an intrinsically impossible
task. Concatenative extraction addresses it without a custom task-specific operator.
Neither old grammar nor its thresholds/results changed. No old horizon result is
rescued and no positive non-myopic claim follows.

This also strengthens the symbolic competitor: both opened behaviors have short
source programs, so their recovery alone would make the LLM ornamental. Next assess
initial-history ambiguity and exact attainable query value under a prospectively
specified source prior, with input/output roles unchanged. If support already
collapses after the initial example, or shallow enumeration suffices, do not fund
an LLM depth sweep on this pair. A meaningful learned proposal contribution must
beat the same short symbolic support on genuinely fresh semantic tasks and pass
joint predictive/transition calibration before planning comparisons. Do not treat
output-conditioned witness search as the policy's prior or use it at deployment.

No paid calls. Live credits/usage/balance remain
245/221.179955339/23.820044661. London Sept9 conservative spend .76167686 includes
the earlier .04 uncertainty; remaining 4.23832314. Automation remains paused;
cluster and historical runtime untouched. Full goal remains incomplete.
