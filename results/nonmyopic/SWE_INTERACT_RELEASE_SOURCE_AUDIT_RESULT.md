# SWE-Interact Release Source Audit Result

Audited: 2026-07-28. This is a zero-call source gate.

## Frozen Source

- Official repository: `https://github.com/scaleapi/SWE-Interact`
- Commit: `b32f98c3b8f76ca65e84341d1f30e5af7135f85d`
- Tree: `d2b16852aa541c780598283ceb0302a4dbe66aff`
- Frozen manifest SHA256:
  `d6184d7decef79e596a522f57b79fecd1aafbc838721290245baaf54d3295102`
- Audit artifact:
  `results/nonmyopic/swe_interact_release_source_audit.json`

The release contains 75 paired multi-turn/single-turn tasks, stratified into
three equal source families. The frozen split keeps 9 mechanics tasks open and
leaves 21 development, 24 confirmation, and 21 retained tasks sealed.

## Result

All preregistered mechanics-source gates pass.

| Source criterion | Required | Observed |
| --- | ---: | ---: |
| Materially incomplete public starts | at least 7/9 | 9/9 |
| At least four verifier-backed hidden atoms | at least 7/9 | 9/9 |
| Dependency or assumption structure | at least 6/9 | 9/9 |
| Reply conditioned on history/workspace | required | pass |
| Generic questions do not dump the checklist | required | pass |
| Forkable private state and target blindness | required | pass |
| External correctness verifier | required | 9/9 |
| No released finite response table | required | pass |

The staged runner is important. Each task's `task.toml` begins with
`steps/01_plan/instruction.md`, a 61--64 word generic instruction that exposes
only the repository and `ask_user` interface. The task-specific top-level
description is not the planning-stage instruction. Each private task block is
135--1,494 words and contains 7--47 conservatively counted verifier-backed
atomic surfaces.

The shared user server passes the full conversation to a generative model. On
review requests it also supplies a fresh snapshot of the committed workspace.
Its disclosure contract starts vague, withholds the hidden checklist, and
raises one relevant issue at a time after inspecting the concrete plan or
implementation. Separate task environments start with empty conversation state,
so identical public histories can be forked while the private task block remains
inside the user server. Final reward comes from released repository tests and
rubrics, not simulated-user approval.

## Decision

**The source gate passes.** SWE-Interact is admitted as an LLM-native candidate
for a separately frozen mechanics serving test. The source establishes a
free-form semantic action space, path-conditioned replies, hidden requirement
structure, and an external endpoint that a finite released table does not
replace.

This does not establish useful non-myopic value. The next gate must show on
mechanics only that:

1. two different concrete roots elicit different valid disclosures from the
   same private initial state;
2. a generic question does not receive equivalent requirement progress;
3. the response depends on the proposed plan or implementation surface;
4. forked repeated calls remain sufficiently stable to compare policies.

Only a passing serving gate may authorize the frozen first-link comparison.
Development, confirmation, and retained task values remain unopened.

## Accounting

- OpenRouter requests: `0`
- OpenRouter cost: `$0`
- OatML, Slurm, SSH, or cluster use: `0`
- Authenticated balance before the next stage: `$9.350490094`
- Reserve: none; the full balance is available by expected scientific value
