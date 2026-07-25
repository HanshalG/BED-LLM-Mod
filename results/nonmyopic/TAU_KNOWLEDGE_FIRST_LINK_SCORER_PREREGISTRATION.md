# tau-Knowledge Target-Blind First-Link Scorer

## Purpose

The V2 opportunity gate passed, but an oracle identified the best roots. This
confirmation tests whether an LLM-native semantic scorer can identify those
roots without required-document labels. It isolates the first link: endpoint
evaluation grants every selected root its oracle-best continuation, so noisy
followup selection cannot create the primary effect.

## Frozen data and model

- Source: `sierra-research/tau2-bench` commit
  `1d244f5dca42944b67a379b44bfeb9f5748f189d`.
- Serving smoke reuses only the two public V2 smoke trees, `task_002` and
  `task_024`.
- Confirmation uses the original untouched 20-task V1 sealed split, in its
  frozen order.
- GPT-5.4, temperature zero, provider reasoning disabled.
- The model sees only the exact scripted first utterance, generated information
  needs, neutralized retrieved-document references, titles, and 700-character
  text excerpts. It never sees full user scripts, required-document IDs,
  endpoint values, evaluation actions, or task notes.

## Frozen policy comparison

For every confirmation task, the flat-schema V2 generator creates five first
queries and four document-conditioned followups per root. Official BM25 top-3
retrieval supplies deterministic transitions.

Two separate scorer calls prevent future leakage into the baseline:

- myopic scorer sees only each root's first query and first results;
- non-myopic scorer sees the complete depth-2 tree and chooses the best shown
  followup for each root.

Both emit independent 0-100 semantic coverage scores for all five roots. Root
selection is deterministic argmax with original-order tie breaking. The primary
endpoint for either selected root is the exact required-document count under
that root's oracle-best continuation. The scorer's selected followup endpoint is
descriptive only.

Ranking fidelity is pairwise accuracy over root pairs with unequal oracle-tail
endpoint values. A tied score receives half credit. The oracle-strength greedy
required-document root remains descriptive because it uses hidden truth and is
not a deployable baseline.

## Gates

Serving smoke is exactly four scorer calls over the existing two trees. It
requires complete flat schemas, exact call count, zero reasoning, and score
variation across roots in both isolated views for each task.

Only a passing smoke releases confirmation. Confirmation is exactly 160 calls:
20 initial generations, 100 followup generations, 20 myopic scores, and 20
non-myopic scores. It passes only if all hold:

- complete trees and scores, exact calls, and zero reasoning;
- structural non-myopic gap on at least 4/20 tasks and mean gap >=0.20;
- at least 50 comparable root pairs;
- non-myopic pairwise ranking accuracy >=0.60;
- non-myopic minus myopic pairwise accuracy >=0.05;
- selected roots differ on at least 4/20 tasks;
- non-myopic root policy wins over myopic on at least three tasks;
- it loses on at most two tasks;
- total required-document advantage is at least two; and
- on structural-opportunity tasks, it selects an oracle-optimal root at least
  50% of the time.

No malformed response, score, task, query, or threshold may be repaired,
replaced, or changed. Failure closes this scorer interface.

## Budget

Before scorer calls, project ledger headroom is `$34.28289803`; live balance
above the protected `$25` Monday reserve is `$34.55727054`. OatML remains
paused. Smoke is projected at `$0.20` with a `$1` cap. Confirmation is projected
at `$3` with an `$8` hard cap.
