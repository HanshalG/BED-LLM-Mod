# tau-Knowledge Focused Receding Continuation

## Purpose

The first-link confirmation found a significant target-blind root-ranking gain,
but the joint full-tree scorer's selected continuations realized only 34 of the
44 required documents available beneath its selected roots. Across all 100
unsealed roots, the joint selector chose an oracle-optimal followup on 65%,
incurring 38 documents of regret.

This experiment changes only the second link. A focused receding scorer receives
one realized root and its four shown continuations per call. It cannot compare
roots or revisit the first decision. Old confirmation trees are development data;
the final split is every remaining untouched no-script tau-Knowledge task.

## Frozen interface

- Source: `sierra-research/tau2-bench` commit
  `1d244f5dca42944b67a379b44bfeb9f5748f189d`.
- GPT-5.4, temperature zero, provider reasoning disabled.
- For each root, the scorer sees the customer opening, initial and refreshed
  information needs, first query/results, four candidate followups/results, and
  neutral document references with titles plus 700-character excerpts.
- It emits four canonical digit-string coverage scores and short rationales.
- Selection is deterministic argmax with original-order tie breaking.
- Required-document IDs, full user scripts, endpoints, and evaluation actions
  are hidden.

The policy controls share the same generated trees and focused continuation
table:

- **non-myopic receding:** full-tree root score, focused followup score;
- **myopic receding:** first-results-only root score, focused followup score;
- **joint non-myopic:** full-tree root score and its original joint followup;
- **random strategy:** seeded random root and followup.

## Splits and stages

Serving smoke uses public `task_018` and `task_008`, scoring all five roots for
exactly 10 calls.

Development uses the already-unsealed 20-task first-link confirmation, scoring
all 100 roots for exactly 100 calls. It cannot support a new policy claim.

Fresh confirmation seed `24337` deterministically orders all 20 remaining
no-script tasks:

`025, 021, 028, 015, 016, 062, 048, 081, 029, 022, 019, 027, 012, 089,
023, 020, 007, 005, 017, 004`.

Confirmation is exactly 280 calls: 20 openings, 20 initial trees, 100
document-conditioned followups, 20 isolated myopic root scores, 20 full-tree
root scores, and 100 focused continuation scores.

## Gates

Smoke requires exact calls, zero reasoning, complete canonical responses, score
variation on at least 8/10 roots, pairwise followup accuracy >=0.55, and at least
7/10 oracle-optimal selections.

Development passes only if all hold:

- at least 200 endpoint-distinct followup pairs;
- pairwise accuracy >=0.60;
- oracle-optimal followup rate >=0.72;
- total all-root regret <=28 documents;
- loss beneath selected non-myopic roots <=5 documents;
- end-to-end non-myopic versus myopic: >=3 wins, <=2 losses, total gain >=3;
- focused non-myopic gains >=5 documents over the original joint selector.

Only a passing development stage releases fresh confirmation. Confirmation
requires:

- complete exact280-call serving and zero reasoning;
- non-myopic root accuracy >=0.60 and gain over myopic >=0.05;
- >=200 comparable followup pairs, focused accuracy >=0.60, optimal rate >=0.70,
  mean regret <=0.30, and selected-root loss <=5;
- versus myopic: >=4 wins, <=2 losses, total gain >=4;
- versus seeded random: >=6 wins, <=4 losses, total gain >=5; and
- focused policy gains >=4 documents over joint continuation selection.

No response, task, query, score, threshold, or malformed output may be repaired,
replaced, or changed.

## Budget

Before calls, project and live headroom above the protected `$25` Monday reserve
is `$32.49855803`. OatML remains paused. Smoke/development/confirmation are
projected at `$0.10/$1.00/$3.50` with hard caps `$0.50/$3/$8`.
