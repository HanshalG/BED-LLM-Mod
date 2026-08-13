# SWE-smith Debug-BED execution mechanics protocol

Date frozen: 2026-08-13

Status: **eight-task zero-model-call mechanics; payloads still unopened**

## Dependency

This protocol depends on pushed commit `88bda876` and the passing V2 source
audit. The exact mechanics cohort is the eight-ID ordered list committed by the
V2 manifest digest. No task may be replaced, reordered, or skipped after its
payload or execution is observed.

## Purpose

Test whether composed SWE-smith defects provide a real two-experiment
diagnostic horizon opportunity under DebugGym. This is an oracle mechanics
gate, not an LLM result. It may use sealed bug patches and test identifiers to
construct and score counterfactual worlds, but none of those values may enter a
later policy prompt or public artifact.

## Private materialization

Only the exact eight mechanics rows may now be read. Their individual IDs,
problem statements, repository names, image names, patches, test names, source
files, traces, and outcomes remain private. Opportunity, development,
confirmation, and reserve rows remain unopened.

For each mechanics task:

1. Start from the released clean repository image.
2. Parse the released composed bug patch as unified diff hunks.
3. Require between 2 and 4 nonoverlapping hunks. Each hunk must apply
   independently and every subset must apply in canonical hunk order.
4. Define one counterfactual latent world for every subset of hunks. The empty
   subset is clean; the full subset is the released true composed-bug world.
5. Candidate experiments are individually addressed released fail-to-pass and
   pass-to-pass tests. Test names remain private and are represented only by
   deterministic opaque aliases during computation.
6. Execute each experiment in every world in two fresh container arms. Reduce
   output to the released parser's categorical test status. Raw output is
   private and discarded after exact replay verification.

The hunk decomposition is an oracle verification device only. It must never be
shown to the later LLM policy or described as the policy's hypothesis space.

## Native debugger handshake

For every structurally eligible task, run two fresh true-world arms and require:

- identical redacted initial evaluation status vectors;
- start PDB through the released DebugGym entrypoint;
- set one breakpoint in a bug-touched function or nearest enclosing callable;
- continue until the breakpoint;
- issue one explicit variable or expression query;
- step once and record the next source location;
- stop and restart PDB with persistent breakpoint restoration;
- exact equality across arms after removing only container ID, absolute
  workspace prefix, elapsed time, memory address, and process ID fields frozen
  in the implementation before execution.

Any unsupported architecture, image pull failure, test timeout, parser error,
patch-application error, nondeterministic status, or debugger mismatch makes
that task mechanically invalid. There are no infrastructure exclusions or task
replacements.

## Exact diagnostic planner

Use a uniform prior over the task's hunk-subset worlds. An experiment response
is its categorical released test status. Likelihoods are deterministic from the
paired replay matrix. Repeated experiments are forbidden.

For belief `b` and experiment `q`, one-step EIG is:

```text
H(b) - E_y[H(b | q, y)]
```

The receding-myopic policy chooses maximum one-step EIG at each state. The
depth-two policy chooses the root maximizing expected entropy reduction after
the best answer-conditioned second experiment. Both policies use the exact
same worlds, experiments, tie-break order, and number of experiment evaluations.

A task has an adaptive dependency only if the depth-two root's optimal second
experiment differs across at least two positive-probability observations. A
task has a positive first link only if:

- the depth-two and myopic roots differ;
- depth-two expected terminal entropy reduction exceeds receding myopic by at
  least `0.05` nats;
- depth-two has no worse expected posterior probability on the full true world;
- at least three valid candidate experiments exist.

## Frozen gates

The mechanics stage passes only if all hold:

1. Source protocol, V2 protocol, manifests, audit, implementation, and pushed
   commit hashes reproduce exactly.
2. Exactly eight frozen mechanics rows and zero other payload rows are read.
3. At least five of eight tasks are structurally eligible.
4. Every structurally eligible task passes all paired status replay and native
   PDB handshake requirements.
5. At least five of eight frozen tasks have an adaptive dependency.
6. At least five of eight frozen tasks have a positive first link.
7. Mean depth-two minus matched-myopic expected entropy reduction across all
   eight tasks, counting invalid/nonqualifying tasks as zero, is at least
   `0.04` nats.
8. No raw or identifying mechanics value is serialized publicly.
9. No model call, patch endpoint, opportunity, development, confirmation, or
   reserve payload is opened.

Failure closes this exact Debug-BED construction. No hunk regrouping, task
replacement, probe addition, threshold change, architecture exclusion, or
post-observation policy repair is permitted.

## Successor if and only if mechanics passes

A pass authorizes a separately frozen low-cost LLM serving gate. The LLM must
generate open semantic bug hypotheses from redacted code/test context, predict
experiment-status likelihoods, regenerate support after each observation, and
rank complete two-experiment paths. Required controls are compute-matched
receding myopic, fixed support, history-blind regeneration, and random. Patch
endpoints remain sealed until serving and opportunity gates pass.

## Accounting

- OpenRouter calls: `0`
- OpenRouter cost: `$0`
- OATML cluster use: none
