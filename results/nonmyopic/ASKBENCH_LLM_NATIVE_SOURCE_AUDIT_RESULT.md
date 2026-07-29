# AskBench LLM-Native Source Audit Result

Status: **failed closed before model calls; AskBench route closed**.

Date: 2026-07-29

Model calls and cost: `0` / `$0`.

Public manifest:
`results/nonmyopic/askbench_llm_native_source_audit/askbench-llm-native-source-audit-20260729/MANIFEST.json`.
SHA-256:
`971a52255b5d8ba134ac825220e3ea11926926f80670fd3ed5eab95e87ae3cbf`.

## Source Findings

Every repository and artifact hash matches the preregistration. The combined
AskMind release contains 400 rows, exactly 100 from each of BBH, GPQA, Math500,
and MedQA. The frozen MedQA filter yields 89 eligible rows, comfortably above
the required 60:

| Required points | Eligible rows |
|---:|---:|
| 3 | 8 |
| 4 | 11 |
| 5 | 38 |
| 6 | 12 |
| 7 | 17 |
| 8 | 3 |

The fixed seed produces disjoint 10-row development and 40-row holdout splits.
All selected rows have the required schema and four answer options.

The released environment also has the desired scientific control flow:

- the candidate sees a degraded question;
- the simulator receives the complete original question and checklist;
- the simulator is instructed to reveal only the answer to the immediate
  clarification;
- the normal loop defaults to three assistant turns, with the final answer
  forced only on the last turn; and
- the judge receives the released expected answer.

## Frozen Gate Failures

Two exact source assertions fail.

First, the candidate payload leakage check is literal. Every MedQA degraded
prompt includes an output-format example of the form `"The answer is A."`.
For rows whose released correct answer is A, that literal equals the hidden
`expected_answer` string even though it is a formatting instruction rather
than answer disclosure. This affects 2/10 development rows and 11/40 holdout
rows; only 37/50 selected rows pass the frozen literal exclusion test.

Second, the evaluator's three-turn default is split across source lines, while
the auditor preregistered an exact one-line source substring. The released code
does default to three turns, but the literal assertion returns false.

These are audit-specification failures rather than scientific defects in
AskBench. Nevertheless, the preregistration made every source gate conjunctive
and explicitly prohibited source-gate repair after aggregate inspection.

## Decision

No initial-world serving call, branch tree, candidate question, simulator
response, final answer, or holdout row is opened. We do not weaken the literal
leakage rule, normalize source formatting, remove answer-A rows, or amend the
auditor after seeing the split.

AskBench is closed for this project. Its 40-row holdout remains untouched, but
the frozen decision rule does not authorize a successor AskBench protocol.
