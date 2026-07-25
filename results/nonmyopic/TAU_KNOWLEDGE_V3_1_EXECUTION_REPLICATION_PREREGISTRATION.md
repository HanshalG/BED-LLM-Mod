# tau-Knowledge V3.1 Execution Replication Preregistration

Frozen on 2026-07-25 before any replication response.

## Purpose And Scope

The original V3.1 holdout passed every ranking and end-to-end gate, but its
unique required-document advantage over myopic was directional: 30 versus 26,
with task-level one-sided `p=.25`. There are no untouched official
tau-Knowledge tasks. This study therefore tests **execution robustness**, not new
task generalization.

The replication reruns the complete frozen policy on the same 20 holdout tasks
with fresh LLM generations under OpenRouter request seed `24395`. Every policy
comparison remains paired within task and generated tree. The two executions
are averaged within task before pooled inference; they are never counted as 40
independent tasks.

## Frozen Protocol

- Official tau2-bench commit
  `1d244f5dca42944b67a379b44bfeb9f5748f189d`.
- Original public artifact SHA-256
  `f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae`.
- Identical V3.1 task IDs, opening generation, eight initial information-need
  hypotheses, five roots, four observation-conditioned continuations per root,
  BM25 top-three retrieval, isolated myopic and full-tree root scorers,
  count-dominant receding continuation scorer, original-order ties, seeded
  random control, required-document endpoint, prompts, and parser.
- GPT-5.4, temperature zero, explicitly non-thinking.
- Exactly 280 physical requests; no repair, reissue, replacement, prompt or
  parser change.
- Existing exact 280-request V3.1 confirmation is the serving qualification;
  no redundant paid serving smoke is needed.
- Projected cost `$3.20`, hard cap `$4.00`.
- OpenRouter only; no OatML cluster use.

## Primary Gates

All must pass:

1. Every original V3.1 confirmation gate passes independently on the fresh
   execution, including root accuracy/gain, continuation accuracy/regret, and
   end-to-end gains against myopic, joint, and random.
2. Exactly 280 adapter responses and HTTP attempts complete with zero retries,
   reasoning tokens, forced exits, or malformed objects.
3. Replication unique required-document coverage is higher than its paired
   myopic control.
4. Averaging the two executions within each of the 20 tasks yields a positive
   non-myopic coverage gain and an exact task-level one-sided sign-flip
   `p<=.05`.
5. Adapter cost is at most `$4.00`.

Failure of the task-clustered significance gate means the endpoint remains
directional even if ranking fidelity reproduces. No third execution or
threshold repair is allowed.

## Secondary Retrieval Family

Before replication outcomes, two standard rank-sensitive endpoints are fixed:

- binary NDCG over the ordered unique documents returned by root then
  continuation;
- reciprocal rank of the first required document.

Replication-alone one-sided task-level sign-flip tests are Holm-corrected as a
two-test family at `.05`. Recall, precision, and F1 are descriptive only.
Secondary passage cannot replace a failed primary coverage gate.

## Analysis

`scripts/tau_knowledge_execution_replication_analysis.py` is frozen before the
run. It uses exact sign flips over 20 task-clustered differences and 100,000
task bootstraps with seed `24396`.

## Budget

Authenticated balance before this registration is `$43.389361484`, leaving
`$18.389361484` above the protected `$25` through-Monday reserve. The `$4` hard
cap preserves at least `$14.389361484` above reserve. The local ledger is
`$86.99544121920736` spent against `$101.14330481920742`.
