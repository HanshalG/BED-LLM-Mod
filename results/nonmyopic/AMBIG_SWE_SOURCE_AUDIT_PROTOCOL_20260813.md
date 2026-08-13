# Ambig-SWE Source Audit Protocol

Date frozen: 2026-08-13

Status: **frozen before opening any Ambig-SWE CSV row content**.

## Objective

Decide whether the official Ambig-SWE release can support a new, genuinely
LLM-native non-myopic BED study. The intended latent state is the missing
requirement set behind an underspecified repository issue. Actions are
free-form clarification questions, observations are replies from the released
user simulator, and the eventual external endpoint is the official executable
SWE-Bench patch test.

This source audit is necessary but not sufficient. It makes no model call,
opens no executable task endpoint, and cannot establish a planning gap or
policy effect.

## Frozen Source

- repository: `https://github.com/sani903/InteractiveSWEAgents`;
- commit: `ed58236332ad039b54f968145d7bed9ba988f262`;
- tree: `a6c2204c8f9e1bfcc4c6d96cc5f5605f980063fa`;
- expected release files:
  - `data/fully-specified.csv`;
  - `data/underspecified.csv`;
  - `data/interaction.csv`;
  - `evaluation/benchmarks/swe_bench/data/full_summaries_verified.xlsx`;
  - `evaluation/benchmarks/swe_bench/interact_run_infer.py`;
  - `evaluation/benchmarks/swe_bench/prompt.py`;
  - `LICENSE`.

The audit records SHA-256 for every expected file and rejects a dirty checkout,
missing file, commit mismatch, or tree mismatch.

## Split Boundary

The audit may first read only CSV headers and task identifiers. It forms the
intersection of task identifiers across the three CSV views, sorts tasks by

```text
SHA256("ambig-swe-20260813:" + task_id), task_id
```

and assigns the first six tasks to `mechanics`, the next 40 to `opportunity`,
the next 64 to `development`, the next 96 to `confirmation`, and every
remaining task to `retained`. The manifest records ordered identifier hashes
and counts, not identifiers or task text.

Only the six mechanics rows may be opened by this audit. Opportunity,
development, confirmation, retained, executable tests, gold patches, and saved
policy outcomes remain sealed.

## Source Gates

All gates are conjunctive:

1. The three CSV views contain the same unique task identifiers, with at least
   300 aligned tasks and nonempty retained split.
2. Hidden and interaction views use a materially shorter initial issue than the
   full view on every mechanics task: at most 80% of full-view whitespace-token
   count and at least 20 full-view tokens omitted.
3. Every mechanics task contains at least three independently recoverable
   missing-information units. A unit is a nonempty numbered or bulleted item in
   a released missing-information field, or a nonempty released missing-detail
   record keyed to the task. The audit does not synthesize units with a model.
4. At least four of six mechanics tasks contain two or more distinct missing
   information categories when the release supplies category labels. If the
   release has no such labels, this gate fails rather than inferring categories
   from task text.
5. The interaction runner gives the user simulator the full issue text and
   dialogue history, but not gold patches, tests, or test outcomes.
6. The simulator accepts unrestricted natural-language questions, preserves a
   per-dialogue history, and can be instantiated independently from the same
   initial task state. A released finite question-to-answer table fails this
   gate.
7. The policy-facing initial observation excludes the full issue, gold patch,
   tests, and hidden endpoint values.
8. Official executable patch tests remain available as an external endpoint
   after policy choices freeze.
9. The license permits research use and redistribution of derived hashes and
   aggregate audit results.

## Decision Rule

A pass authorizes only a separately frozen, zero-call horizon-opportunity
construction on the 40-task opportunity split. That construction must define a
source-verifiable or externally audited first-action tradeoff before any LLM
policy call. It must also freeze:

- a semantic answer-obedience and repeated-fork stability smoke;
- an explicit dynamic-support policy;
- a compute-matched myopic control using the same generated support budget;
- a random-question control;
- common-random-number user replies where the simulator permits them; and
- a target-blind proxy endpoint before full patch execution is considered.

A failure closes this exact release-derived route. No threshold, cohort size,
field interpretation, or source commit may be changed after mechanics content
is opened.

## Accounting

OpenRouter calls and cost: `0`. OATML/Slurm/SSH use: `0`.
