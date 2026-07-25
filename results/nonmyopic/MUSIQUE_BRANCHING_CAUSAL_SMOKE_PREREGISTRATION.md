# MuSiQue 4-Hop Branching Causal Smoke Preregistration

Date frozen: 2026-07-25

## Purpose

Test whether an LLM-generated, paragraph-conditioned dependency belief can
identify the deep root in the prospectively qualified MuSiQue `4hop3`
environment.

The zero-call train audit established the external opportunity. This smoke
tests the first deployability link on two frozen development tasks. It is not
a policy result or holdout claim.

## Frozen Tasks

The official train source, seed-24353 split, and hashes are fixed in
`MUSIQUE_BRANCHING_UNLOCK_AUDIT_PREREGISTRATION.md`.

Use the first two IDs in the frozen 40-task development order:

1. `4hop3__405751_4520_37620_55609`
2. `4hop3__360282_544708_764770_54023`

These IDs were selected before their question, answer, decomposition, support
indices, titles, or paragraphs were accessed. The other 38 development and all
240 holdout endpoints remain sealed.

## Exact Sixteen-Call Interface

Model: `openai/gpt-5.4`, temperature zero, explicit non-reasoning.

For each of two tasks:

1. One initial call sees only the final question. It emits eight distinct
   natural-language dependency-chain hypotheses, six distinct corpus search
   queries, and one immediate 0--100 usefulness score per query.
2. Deterministic BM25 executes each query against the sealed 20-paragraph
   context. Titles are hidden before retrieval. Original paragraph order
   breaks ties.
3. Six independent refresh calls each see the final question, initial
   hypotheses, root query, and exactly one retrieved title and paragraph.
   Each emits eight refreshed unresolved dependency hypotheses and four
   follow-up queries, with no score.
4. BM25 executes each follow-up while excluding the first document.
5. One state-only continuation call per task scores all 24 follow-ups under
   three states:
   - the correctly aligned refreshed hypotheses;
   - the cyclic next-root refreshed hypotheses;
   - the unchanged initial hypotheses.

The continuation scorer never sees the original question, root query, root
title, root paragraph, answer, decomposition, support indices, annotations, or
branch role. It receives only each belief state plus neutral follow-up IDs,
query strings, and retrieved second-document titles.

Total: `2 * (1 + 6 + 1) = 16` physical requests.

## Blinding

Seed `24354` freezes whether aligned beliefs are labeled A or B:

```text
task 1 roots: B A B A B A
task 2 roots: A B A B A B
```

State C is always the unchanged initial state. Labels are globally and
within-task balanced. Scores may be JSON integers or canonical decimal digit
strings from 0 through 100. Responses must be exact flat JSON with no repair,
trailing extraction, reissue, or scientific retry.

## Frozen Policies

All policies share generated queries, BM25 results, hypotheses, and scorer
calls. Ties use earlier generated order.

- **Myopic receding:** root with maximum initial immediate score; aligned
  follow-up argmax for that realized root.
- **Fixed-support d2:** immediate root score plus maximum state-C continuation;
  state-C follow-up argmax.
- **Model-aware d2:** immediate root score plus maximum aligned refreshed-state
  continuation; aligned follow-up argmax.
- **Shuffled-belief d2:** immediate root score plus maximum cyclic-state
  continuation; cyclic-state follow-up argmax.
- **Random receding:** seeded root (`24355` plus task index), then aligned
  follow-up argmax.

No policy receives an annotated role.

## External Endpoint

Map retrieved paragraph indices to the hidden four-step decomposition only
after all model responses freeze.

- A selected pair containing deep-root support step 1 followed by deep-child
  support step 2 has connected-prefix length 2.
- Any other pair containing at least one annotated support has length 1.
- A pair containing no annotated support has length 0.

Also record whether generated roots retrieve the deep and shallow supports and
whether the aligned continuation after a deep-root retrieval selects the deep
child.

## Gates

The smoke passes only if all hold.

### Mechanics

- exactly 16 physical requests and 16 HTTP attempts;
- zero transport retries, reasoning tokens, forced exits, parse failures,
  repairs, or scientific retries;
- all initial and refreshed states contain eight unique nonempty hypotheses;
- all root/follow-up query lists have the exact size and unique nonempty text;
- each task retrieves at least three distinct first documents;
- each task's generated roots include both the deep and shallow support;
- every refreshed state differs from the task's initial state and at least
  five of six refreshed states are pairwise distinct per task;
- A/B labels are balanced;
- aligned continuation scores vary on at least 10/12 roots;
- aligned and shuffled vectors differ on at least 8/12 roots;
- aligned and initial vectors differ on at least 8/12 roots;
- cost is at most `$0.50`.

### Scientific Signal

- on at least one task, an aligned deep-root continuation retrieves the
  annotated deep child;
- model-aware d2 selects a deep-root candidate on at least one task;
- model-aware connected-prefix length is at least myopic on both tasks and
  strictly greater on at least one;
- model-aware connected-prefix length is at least fixed-support and
  shuffled-belief d2 on both tasks.

This is conjunctive. Failure closes this exact higher-hop causal interface
before the remaining development or holdout endpoints are accessed. Passing
authorizes a separately frozen multi-task development ranking gate.

## Budget

- Projected cost: `$0.20`.
- Hard run cap: `$0.50`.
- Weekend incremental-spend target remains at most `$15`; preserve `$25`.
- OatML is paused; no cluster job is involved.
