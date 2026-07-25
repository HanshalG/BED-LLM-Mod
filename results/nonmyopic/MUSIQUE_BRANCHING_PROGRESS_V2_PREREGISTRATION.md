# MuSiQue 4-Hop Discrete-Progress V2 Smoke Preregistration

Date frozen: 2026-07-25

## Motivation And Scope

V1 cleanly generated deep-root-to-deep-child continuations on both tasks, but
its arbitrary immediate-plus-continuation 0--100 scores had pooled root
fidelity `rho=.073` and never improved over myopic. This V2 makes one
prospective objective change: score current and terminal dependency progress
on the same discrete 0--4 scale.

V1 remains failed. V2 uses fresh tasks and cannot rescue or replace it.

## Frozen Tasks

Use development positions 3 and 4 from the seed-24353, hash-locked train split:

1. `4hop3__493923_300471_583182_62462`
2. `4hop3__21483_551941_773286_24137`

Only IDs were accessed before this preregistration. Their question, answer,
decomposition, support indices, titles, and paragraphs remain unread. The
other 36 development and all 240 holdout endpoints remain sealed.

## Shared Tree Generation

Model, temperature, BM25 execution, six roots, eight initial hypotheses, six
isolated paragraph-conditioned refreshes, eight refreshed hypotheses, four
follow-up queries, blinding, parsing, and exact 16-call schedule remain as in
V1.

The initial generator no longer emits or sees an immediate utility score. This
removes the failed cross-stage arithmetic rather than rescaling it.

## Discrete Progress Scorer

One state-only scorer per task receives aligned, cyclically shuffled, and
unchanged initial hypotheses plus follow-up query strings and retrieved titles.
It still sees no question, root query, root result, answer, decomposition,
support role, or annotation.

For every root/state it outputs:

- `current_progress`: one integer band before a follow-up;
- `terminal_progress` for each of four follow-ups.

Frozen bands:

```text
0 = no concrete dependency in the final question is resolved
1 = one useful entity or fact is resolved
2 = a connected two-step dependency chain is resolved
3 = that connected chain plus the independent root is resolved
4 = the final relation/answer is resolved from all required dependencies
```

These are model predictions from natural-language beliefs, not annotated
labels. The exact hidden connected-prefix endpoint remains 0/1/2 after two
retrievals.

## Blinding And Controls

Seed `24356` freezes aligned labels:

```text
task 1 roots: A B A B A B
task 2 roots: B A B A B A
```

Random roots use seed `24357` plus task index.

- **Myopic receding:** aligned `current_progress` argmax, then aligned terminal
  argmax for the selected root.
- **Model-aware d2:** maximum aligned `terminal_progress` over follow-ups.
- **Fixed-support d2:** maximum state-C terminal progress.
- **Shuffled-belief d2:** maximum cyclic-state terminal progress.
- **Random receding:** seeded root, then aligned terminal argmax.

All ties use earlier generated order.

## Gates

### Mechanics

- exact 16 physical requests and HTTP attempts;
- zero retries, reasoning tokens, forced exits, parse failures, repairs, or
  scientific retries;
- exact unique hypothesis/query cardinalities;
- at least three distinct first documents per task;
- generated roots retrieve both deep and shallow supports on each task;
- every refreshed state differs from initial and at least five of six are
  pairwise distinct per task;
- balanced A/B labels;
- aligned terminal vectors vary on at least 10/12 roots;
- aligned-vs-shuffled and aligned-vs-initial terminal vectors differ on at
  least 8/12 roots;
- cost at most `$0.50`.

### Scientific

- an aligned deep-root continuation retrieves the deep child on at least one
  task;
- model-aware selects a deep-root candidate on at least one task;
- pooled Spearman between predicted aligned terminal progress and exact best
  reachable root prefix is at least `.30`;
- model-aware exact prefix is at least myopic on both tasks and strictly
  greater on at least one;
- model-aware exact prefix is at least fixed-support and shuffled-belief d2 on
  both tasks.

Failure closes this discrete-progress construction. Passing authorizes a
separately frozen multi-task development ranking gate, not holdout access.

## Budget

- Projected cost `$0.20`; hard cap `$0.50`.
- Weekend incremental target remains at most `$15`; preserve `$25`.
- OatML remains paused.
