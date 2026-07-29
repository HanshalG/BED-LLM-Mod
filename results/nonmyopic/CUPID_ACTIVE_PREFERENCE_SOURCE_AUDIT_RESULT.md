# CUPID Active-Preference Source Audit Result

Status: **all zero-call source gates pass; serving smoke authorized**.

Date: 2026-07-29

Model calls and cost: `0` / `$0`.

Public manifest:
`results/nonmyopic/cupid_active_preference_source_audit/cupid-active-preference-source-audit-20260729/MANIFEST.json`.
SHA-256:
`2f742ddbacd64eade99fa0148b3c5a11a8676f5cfceb5e666f0966fa6785f790`.

## Released Source

The audit binds CUPID repository commit
`a8560cab293ae98be4fe260689d58bddf96b51ef`, its source tree, the
Hugging Face dataset revision, the downloaded parquet, and the official
formatter, evaluator, preference-inferrer, and inferrer-prompt hashes.

All 756 released rows pass the structural audit. The dataset is exactly
balanced across 252 `consistent`, 252 `contrastive`, and 252 `changing`
instances. Every row has:

- a nonempty current request, context, contextual preference, and checklist;
- exactly eight valid prior dialogue sessions;
- at least one prior session for the current context; and
- at least two sessions from other contexts.

Checklist sizes are 2 items for 257 rows, 3 for 375 rows, and 4 for 124 rows.
All `(persona_id, instance_type)` keys are unique.

## BED Adaptation

This is an **active contextual-preference interview derived from CUPID**, not
the official CUPID evaluation protocol.

The candidate policy initially receives:

- the current request and context factor; and
- the two earliest prior dialogues whose context differs from the current
  context.

It does not receive the released current preference, checklist, structured
preference metadata from prior sessions, same-context dialogue, or remaining
dialogues. The official CUPID formatter was also executed on sentinel data:
it emits user/assistant dialogue and omits context and preference metadata.

The intended hidden state is the released open-text current contextual
preference. A future target model may answer generated binary clarification
questions while conditioned on that hidden state. The planner must generate
and path-dependently refresh open-text preference hypotheses and their
predicted answer signatures. CUPID's released checklist remains an external
terminal endpoint rather than an input to planning.

## Frozen Split

Seed `37400` gives a deterministic, disjoint split:

| Split | Consistent | Contrastive | Changing | Total |
|---|---:|---:|---:|---:|
| Serving smoke | 1 | 2 | 2 | 5 |
| Development | 5 | 5 | 5 | 15 |
| Holdout | 20 | 20 | 20 | 60 |
| Unused | 226 | 225 | 225 | 676 |

The public manifest contains only source IDs, row hashes, type labels, and
structural counts. It does not contain preferences, checklists, requests, or
dialogues.

## Decision

All nine conjunctive source gates pass. The exact five-case, ten-request
planner/target serving smoke may be specified and preregistered. No
development or holdout policy endpoint has been viewed, and passing this
source audit alone makes no efficacy claim.
