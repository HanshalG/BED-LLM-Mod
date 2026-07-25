# tau-Knowledge Count-Dominant Receding Continuation V3

## Motivation

V2's evidence-only interface improved branch ranking substantially, but failed
closed on two verbose responses and did not improve the selected-root policy
gates. The selected-root audit showed that free-form utility scores traded one
urgent document against multiple distinct useful documents and sometimes
overweighted a refreshed product-specific belief.

V3 is the final continuation development interface. It uses the LLM for the
irreducible semantic operation: deciding which returned documents support
plausible unresolved needs generated along the realized path. Selection then
uses a frozen count-dominant score.

## Frozen interface

- Tasks, trees, model, temperature, document excerpts, root scorers, controls,
  endpoint calculations, and task splits are unchanged.
- Candidate query strings, required-document IDs, full scripts, and endpoints
  are hidden.
- The opening and initial information needs define the primary objective.
  Refreshed information needs remain visible but are explicitly fallible; they
  cannot erase original goals, comparisons, named products, or parallel
  requests.
- For each candidate, GPT-5.4 counts distinct new returned documents whose
  supplied title or excerpt materially supports at least one plausible
  unresolved need. Already acquired, duplicate, wrong-product, surface-match,
  and imagined-intent documents do not count.
- Scores use nonoverlapping count bands: 0 useful documents maps to 0-9, 1 to
  30-39, 2 to 60-69, and 3 to 90-99. The final digit only breaks ties by
  directness and breadth. One additional useful document always dominates.
- Output is exactly four canonical digit-string scores with no rationales.
  Original-order argmax tie breaking is unchanged.

## Stages and gates

The exact V1/V2 gates remain frozen.

Smoke uses the same two public tasks and all five roots for exactly 10 calls. It
requires complete canonical responses, zero reasoning, score variation on at
least 8/10 roots, pairwise accuracy at least 0.55, and at least 7/10
oracle-optimal selections.

Development uses all 100 already-unsealed roots for exactly 100 calls. It
requires at least 200 comparable pairs, accuracy at least 0.60, at least 72/100
optimal selections, regret at most 28, selected non-myopic-root loss at most 5,
non-myopic versus myopic at least 3 wins / at most 2 losses / total gain at least
3, and improvement over the original joint selector at least 5.

Only a complete passing development result releases the untouched 20-task,
280-call confirmation selected with seed `24337`. Confirmation gates remain
exactly those in V1. There will be no V4 on these development trees.

## Budget

Live balance before V3 is `$55.259433`, leaving `$30.259433` above the protected
`$25` Monday reserve. Compact smoke, development, and conditional confirmation
are projected at `$0.08`, `$0.80`, and `$3.20`, with hard caps of `$0.50`, `$3`,
and `$8`. OatML remains paused.
