# SWE-bench Lite Bug-Localization Unlock Result

## Decision

The frozen zero-call opportunity audit failed. This deterministic
issue-to-code retrieval construction is closed before any LLM call or
SWE-bench Lite test endpoint access.

## Results

All nine no-direct-leak development issues reproduced at their exact base
commits and exposed one changed Python file. Every issue had at least three
distinct root queries, but only two of the eight opportunity/diversity gates
passed:

- mean distinct root top-1 files: `2.6667`, below `3.0`;
- immediate root coverage: `5/9`, within the at-most-`6/9` gate;
- positive pair gains: `1/9`, below `3/9`;
- mean pair gain: `.1111`, below `.33`;
- oracle first-root changes: `1/9`, below `2/9`;
- positive non-myopic gaps: `1/9`, below `2/9`; and
- mean non-myopic gap: `.1111`, below `.22`.

The sole positive case was `pydicom__pydicom-1139`. Its immediate roots did
not retrieve `pydicom/valuerep.py`, while followups under two non-greedy roots
did. This produced pair gain `1`, an oracle-root change, and non-myopic gap
`1`.

Five issues already retrieved the target in a root top three, so no second
step could improve their binary endpoint. Of the other four, three remained
unreachable under every frozen code-IDF followup. The construction therefore
has neither broad delayed reachability nor enough root-dependent continuation
value to justify semantic LLM scoring.

## Interpretation

This result does not test whether an LLM can reason about software faults.
It tests the prerequisite structural claim that the frozen file-retrieval
interface exposes enough two-step value for a non-myopic semantic policy to
measure. That prerequisite fails.

No tokenizer, path weight, top-k, followup generator, repository, threshold,
or cohort was changed after outcomes. No fresh-test problem statement, patch,
changed-file endpoint, or test patch was read. No prompt repair, LLM smoke, or
holdout run follows.

## Budget

- API calls: `0`.
- OpenRouter spend: `$0`.
- Protected reserve affected: no.
- OatML cluster use: none.

## Artifacts

- Preregistration:
  `results/nonmyopic/SWEBENCH_LITE_BUG_LOCALIZATION_UNLOCK_PREREGISTRATION.md`
- Full development audit:
  `results/nonmyopic/swebench_lite_bug_localization_unlock/OPPORTUNITY.json`
- Dataset revision:
  `69611d31007e1c6731db8bd5b5c3f2d33f5bab6e`
