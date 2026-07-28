# SWE-Interact Mechanics Serving Preregistration

Date: 2026-07-28

**Status: frozen before any SWE-Interact model request.**

## Claim Under Test

The zero-call source audit established that SWE-Interact has a private,
history-conditioned user simulator and external task verifiers. This serving
gate tests whether the released disclosure contract is realized by an actual
model:

> From one shared initial reply, different concrete clarification roots should
> elicit different task-valid hidden requirements; a generic checklist request
> should not receive equivalent progress; and review feedback should change
> with the inspected implementation surface.

This is mechanics only. It does not select a policy, estimate EIG, open
development tasks, or claim that depth two beats myopic selection.

## Frozen Source And Tasks

- Official source: `https://github.com/scaleapi/SWE-Interact`
- Commit: `b32f98c3b8f76ca65e84341d1f30e5af7135f85d`
- Tree: `d2b16852aa541c780598283ceb0302a4dbe66aff`
- Manifest SHA256:
  `d6184d7decef79e596a522f57b79fecd1aafbc838721290245baaf54d3295102`
- Source-audit artifact:
  `results/nonmyopic/swe_interact_release_source_audit.json`

Use exactly one already-open mechanics task per source family:

1. `deepswe_clack-async-autocomplete-options`
2. `rf_task-694b4b99829f00e24fd118a1`
3. `swebenchpro_instance_qutebrowser__qutebrowser-fea33d607fde83cf505b228238cf365936437a63-v9f8e9d96c85c85a605e382f1510bd08563afc566`

Development, confirmation, and retained task values remain sealed.

## Models And Reasoning

- Released-user role: `openai/gpt-5.4` through OpenRouter, high reasoning,
  temperature zero. High reasoning belongs to the environment simulator and
  follows the released SWE-Interact server configuration.
- Semantic requirement annotator: `openai/gpt-5.4-mini`, reasoning disabled,
  temperature zero.
- BED policies are absent from this gate. In later policy experiments BED
  remains nonreasoning; only the naive baseline may use thinking.

Both adapters use seed `24423`. No semantic repair, reissue, fallback, or
post-parse coercion is allowed. Bounded transport retries are logged separately.

## Fork Protocol

For each task:

1. Call the user simulator once with the released private persona and a request
   for the short version of the task.
2. Freeze that exact initial reply.
3. Fork seven branches from the identical history:
   - one generic request for the complete checklist;
   - two identical calls to targeted root A;
   - two identical calls to targeted root B;
   - one review branch with implementation surface A;
   - one review branch with implementation surface B.
4. The review branches use the exact private automatic-snapshot message shape
   from the released server and frozen compact diff-like summaries. They test
   response conditioning, not repository correctness.
5. In one independent call per task, annotate `INITIAL`, `GENERIC`,
   `ROOT_A_1`, `ROOT_A_2`, `ROOT_B_1`, `ROOT_B_2`, `REVIEW_A`, and
   `REVIEW_B` against the released verifier-backed requirement catalog.

The annotator counts only content asserted, confirmed, or corrected by the
simulated maintainer. It excludes details that appear only in the agent's
question. Branch labels are requirement IDs local to a task; private text and
raw replies are never written to the public result.

## Exact Accounting

- Released-user calls: `3 + 3 * 7 = 24`
- Semantic annotator calls: `3`
- Total logical calls: `27`
- User concurrency: at most `24`
- Maximum transport retries: `3` per adapter
- Maximum output: `4,096` tokens for user replies and `1,024` for annotations
- Projected cost: `$1.50`
- Fail-closed stage ceiling: `$4.00`

The ceiling prevents a runaway provider response. It is not a reserve or an
allocation rule: the full authenticated balance remains available for later
stages according to expected scientific value.

## Frozen Gates

All are conjunctive:

### Integrity

1. exactly 24 released-user requests and 3 annotator requests;
2. HTTP attempts equal logical requests plus logged transport retries;
3. no empty reply, forced exit, or semantic repair;
4. annotator reasoning tokens equal zero;
5. combined cost is at most `$4.00`;
6. every annotation has exactly the eight expected unique keyed rows and only
   valid local requirement IDs.

### Semantic Mechanics

1. `INITIAL` contains at most one concrete requirement on all `3/3` tasks;
2. `GENERIC` contains zero newly disclosed requirements on all `3/3` tasks;
3. both targeted roots disclose at least one preregistered intended
   requirement, and their annotation sets differ, on all `3/3` tasks;
4. repeated identical root forks have exact requirement-set agreement on all
   `6/6` task-root pairs;
5. both review surfaces elicit their preregistered intended correction and the
   two review annotation sets differ on all `3/3` tasks.

## Decision

A pass authorizes a separately committed first-link mechanics protocol. That
protocol must generate a shared semantic hypothesis/action bank with a
nonreasoning BED model and compare matched-compute myopic versus depth-two
selection before any development efficacy run.

Any failed semantic gate closes this exact simulator/model/interface route.
Transport failure may be retried only through the frozen bounded retry policy.
No task substitution, threshold relaxation, parser repair, or selective rerun
is allowed.
