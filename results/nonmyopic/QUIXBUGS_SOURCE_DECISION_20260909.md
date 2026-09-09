# QuixBugs contract: useful reference, weak primary candidate

Pinned upstream4257f44b0ff1181dedaedee6a447e133219fcebf. Read only README, license,
driver and Git tree metadata; no task program, corrected implementation or test
label contents loaded. No upstream code imported/executed. Banked audit hashes:
README021ddf80, license3bb03003, driver47b5f2f1; inventoryedb90b30.

The [upstream repository](https://github.com/jkoppel/QuixBugs/tree/4257f44b0ff1181dedaedee6a447e133219fcebf)
describes40single-line algorithm defects with corrected Python implementations.
MIT license text was checked. Inventory has50Python files per buggy/correct
directory, including tests/helpers, not50distinct tasks;31JSON banks and42Python
test/helper files. The [original paper](https://jkoppel.github.io/QuixBugs/quixbugs.pdf)
describes the small algorithm-repair setting. A
[2021 Codex evaluation](https://arxiv.org/abs/2111.03922) already found competitive
repair ability, but that is not evidence of this project's current model accuracy.

## Observation contract

The stock tester prints the full input/expected-output record and corrected output,
then compares defective code. It loops the released bank, not a budgeted single
oracle query. Python implementations are imported in-process; Java subprocess
output is read without a timeout in this driver. This is not an isolated executor
for untrusted generated patches. Nine graph-based routines use a separate driver
path. A benchmark adapter would need explicit per-query access, input domains,
normalization of exceptions/generators, isolation/resource bounds and disjoint
target labels. Wrapping the current full driver would leak more observations than
the planner paid for. The audit itself never executes that driver.

## Decision

Do not launch paid QuixBugs work or build a full adapter yet. Deprioritize it as the
main route because familiar, small single-line algorithms are at substantial risk
of prior-knowledge saturation. This is an inference from task structure and prior
literature, NOT a measured horizon null or a permanent claim of impossibility.
Released correct code offers coherent reference behavior but does not establish
remaining uncertainty, good LLM predictive weights or non-myopic query advantage.

Project evidence already includes closed SWE-bench Lite retrieval and HumanEval
disambiguation opportunities; neither can be reopened by renaming it debugging.
Before building another code environment, prefer a source-contract audit of real
multi-function bugs (e.g. BugsInPy): specification quality, reproducible isolated
reference versions, task-family splits, allowed observations and computational
cost must be established. A new source must demonstrate non-saturated ambiguity
and LLM hypothesis value before a depth solver. Do not manufacture an unlock by
hiding freely available tests or restricting broad queries after outcomes.

Two offline tests.08s/lint: AST analysis never executes source, inventory rejects
truncation and reads paths only. Audit modelcalls0, cost0;four audit network reads
plus exploratory primary-source reads. Account remains245/221.306531939/23.693468061,
London conservative remaining4.11174654. Previous turn was a completed reasoning
comparison; current turn adds pinned source-contract evidence and avoids another
unqualified benchmark build. Full research goal remains unachieved.
