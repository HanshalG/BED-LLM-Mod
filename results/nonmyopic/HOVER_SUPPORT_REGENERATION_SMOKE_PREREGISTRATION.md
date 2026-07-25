# HoVer Support-Regeneration Smoke Preregistration

## Purpose

Test whether an LLM can perform the irreducible part of non-myopic semantic
BED on already-open HoVer opportunity tasks:

1. generate a weighted open-world evidence-chain belief;
2. regenerate that belief after each possible first document;
3. expose genuinely new true-support coverage in the regenerated state; and
4. assign higher future uplift to roots whose regenerated state supports a
   better exact three-document path.

This is prompt and serving development on two already-open robust-opportunity
tasks. It is not a fresh efficacy test and cannot enter the paper as
confirmation.

## Frozen Public Inputs

- Endpoint-free fixture SHA-256:
  `86caf82219fcdbeddcec8462cc225458d3582bd8b438fd15af1ecd43a0efabce`.
- Tasks, in order:
  - `a88d2342-f506-4b15-8578-fb7861eb54c1`;
  - `3cc79319-433d-49b2-97f3-953ba925d6bd`.
- Each fixture row exposes only the claim, hop count, and official top-100
  TF-IDF title catalog.
- The first ten titles are root candidates.
- Root observations are exact article text from the official Wikipedia
  database SHA
  `c37ee397916ec0bffacfe8902db454a5cda88a7a188409217b2e15231fe5ee2f`,
  truncated deterministically to 2,400 normalized characters.
- Model: `openai/gpt-5.4`, nonreasoning, temperature 0.
- Initial and regenerated beliefs contain eight weighted free-text semantic
  evidence-chain hypotheses. Weights are model supplied 0--100 scores and are
  normalized only for diagnostics.
- Every regeneration proposes three distinct next-document IDs from the
  top-100 catalog.

No label, support title, support sentence, exact path, or endpoint value enters
generation or scoring.

## Frozen Flat Grammars

No JSON is requested.

- Initial response: exactly eight `Hnn|weight|hypothesis` lines followed by one
  `Rnn|score` line per root.
- Regeneration response: exactly eight hypothesis lines followed by exactly
  three `Nnn|Cxxx` catalog proposals.
- Future scorer response: exactly one `Rnn|score` line per root.

Any extra text, missing line, duplicate hypothesis/proposal, out-of-range
integer, unknown catalog ID, or repair attempt fails the stage.

## Contingent Stages

### Serving

Use the first task, its first eight roots, and exactly ten model calls:

- one initial belief/direct-score call;
- eight independent root regenerations;
- one aligned future-uplift scorer.

Pass requires exact request accounting, zero retries/reasoning/forced exits,
all parsing, every regenerated state distinct from the initial state, at least
seven pairwise-distinct regenerated states, valid proposals, a nonconstant
future score vector, and cost at most `$0.15`.

No endpoint file is loaded in this stage. Passage authorizes the mechanics
stage. Failure authorizes only a separately committed format/prompt revision
on these same open inputs.

### Mechanics

Use both tasks, all ten roots, and exactly 28 calls:

- two initial calls;
- twenty independent root regenerations;
- six future scorers: aligned, cyclically shuffled, and fixed-support for each
  task.

The cyclic shuffle is a one-root rotation within each task. The fixed-support
condition retains the initial belief for every root and supplies only the
target-blind exact title mentions exposed by that root. The aligned condition
uses the matching regenerated belief and its proposed next titles.

Freeze all raw generations and scores before loading HoVer supporting facts or
recomputing exact root values.

## Frozen Policies and Endpoints

- `myopic`: maximize the initial direct score.
- `regenerated_d3`: maximize initial direct score plus aligned future uplift.
- `fixed_support`: maximize initial direct score plus fixed-support uplift.
- `shuffled_future`: maximize initial direct score plus cyclically shuffled
  uplift.
- `random`: seeded root selection with seed `24406 + task_index`.

The primary external endpoint is exact `V3(root)`: best supporting-document
coverage reachable from the selected root under the preregistered title-link
transition. Continuation execution is secondary.

Mechanism diagnostics:

- exact support-title coverage in initial versus regenerated hypothesis text
  and proposed titles;
- whether the depth-3 oracle root's regeneration newly covers true support;
- whether its first exact oracle continuation appears among the three
  proposals;
- pairwise accuracy between future/full scores and exact root values.

## Mechanics Gates

All are conjunctive:

1. exact 28 physical requests and HTTP attempts;
2. zero transport retries, reasoning tokens, and forced exits;
3. every response parses without repair;
4. all 20 regenerated states differ from their initial states;
5. at least nine pairwise-distinct regenerated states per task;
6. aligned future vectors are nonconstant and differ from both shuffled and
   fixed vectors on both tasks;
7. the oracle-root regenerated state increases exact support coverage over
   the initial state on at least one task;
8. the oracle root proposes its exact first continuation on at least one task;
9. `regenerated_d3` changes the myopic root on at least one task;
10. mean exact selected `V3` is strictly above myopic and at least both
    fixed-support and shuffled-future;
11. `regenerated_d3` selects an exact oracle root on at least one task;
12. pooled aligned full-score pairwise accuracy against exact root values is at
    least `0.55`;
13. cost is at most `$0.50`.

This is a developmental conjunction. A pass authorizes a separately frozen
fresh-development validation; a failure diagnoses the link and may motivate a
new version only on these same open tasks. It never authorizes selecting a
favorable fresh subset after scores are seen.

## Budget

The serving stage projects below `$0.08`; mechanics projects below `$0.25`.
The local OpenRouter ledger cap remains `$105`, the protected `$25` reserve is
unchanged, and OatML is prohibited.
