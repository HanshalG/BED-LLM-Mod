# HiL-Bench Progressive-Support Opportunity Result

Run on 2026-07-25 against official commit
`352d14c861f2531949dfa91848d4b2fe46b8a247`.

## Result

**All frozen structural gates passed.**

| Metric | Result | Gate |
| --- | ---: | ---: |
| usable opportunity tasks | 40 | exactly 40 |
| blockers per task | 3--5 | 3--5 |
| tasks with question + business-info blockers | 40 | at least 34 |
| tasks with all three blocker sources | 39 | at least 30 |
| business-info blockers | 50 | at least 40 |
| business blockers with novel observed evidence | 48/50 (`.96`) | at least `.60` |
| mean novel blocker-token recall | `.169986` | at least `.03` |

The opportunity cohort contained 67 question blockers, 50 business-information
blockers, and 39 schema blockers. Tasks had a mean 29.5 business-information
documents.

## Interpretation

HiL-Bench supplies a released progressive-discovery mechanism suitable for the
project's LLM-native hypothesis: environment inspection exposes
blocker-relevant content absent from the initial request, while official
blockers and resolutions remain external endpoints. This is substantially
stronger structure than the independent clarification slots in EComAgentBench
or the eventually revealed intents in pi-Bench.

It is not a non-myopic planning result. The blockers are independent, and a
lexically relevant observation does not prove that an LLM will generate a valid
new blocker question, that a target-blind scorer can predict that gain, or that
spending a turn on inspection beats asking immediately.

The result authorized only the preregistered ten-request support-expansion
smoke. Cost was `$0`; no OpenRouter request and no OatML cluster resource was
used.

Artifact:
`results/nonmyopic/hil_bench_progressive_opportunity/hil-progressive-opportunity-20260725T160000Z/audit.json`.
