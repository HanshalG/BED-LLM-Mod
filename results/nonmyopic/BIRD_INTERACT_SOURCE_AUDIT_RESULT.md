# BIRD-Interact Source Audit Result

Audited: 2026-07-26. This is a zero-call source gate.

## Pinned Sources

- Official repository: `https://github.com/bird-bench/BIRD-Interact`
- Repository commit: `451fe2c3518ee1cf908d8139e2913483bd519381`
- Mini-Interact dataset:
  `https://huggingface.co/datasets/birdsql/mini-interact`
- Dataset revision: `10253f235dbc6092a97e9c0acd30dba6203d8e59`
- `mini_interact.jsonl` SHA256:
  `32eea6778d69f4a052622a9981ee95cc5dddbff62d0d65ff6e06bd5bec757087`

## What Is Released

The 300-task SQLite release is a real interactive semantic environment. It
contains 733 critical ambiguity annotations over 26 databases, with 126 tasks
having at least three critical ambiguity points. The official user simulator
maps an arbitrary clarification question to an ambiguity or SQL segment and
then generates a natural-language answer conditioned on the task's hidden
ground-truth SQL.

The public Mini-Interact task file nevertheless contains:

- zero nonempty `sol_sql` fields;
- zero nonempty `test_cases` fields;
- zero follow-up tasks;
- no alternative intent or SQL worlds;
- no prior over interpretations;
- one annotated SQL interpretation per ambiguity term.

Repeated ambiguity labels do not repair this. The audit finds 15 normalized
`(database, term)` pairs with distinct SQL snippets, but all occur across
different task records. Within a task, zero terms have multiple released
interpretations. These are corpus repetitions, not a task-level prior or
world-conditioned response map.

## Decision

**Direct BIRD-Interact replay is not authorized as a BED experiment.** The
benchmark is interactive, but each task has one canonical hidden intent rather
than a released family of mutually exclusive latent worlds. Treating
LLM-generated guesses as if they were benchmark alternatives would change the
environment.

There is a distinct construction worth preserving: generate several coherent
SQL-intent worlds prospectively, sample the hidden world from a frozen prior,
and use BIRD's clarification simulator and executable SQLite endpoint as the
observation and validation machinery. That would be a new BIRD-derived
semantic BED environment, not a direct benchmark result. It must first pass a
zero-cost manifest gate and then an exact, capped mechanics smoke before any
efficacy study.

OpenRouter spend: `$0`. OatML use: none.
