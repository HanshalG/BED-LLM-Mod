# tau-Knowledge Future-Subtree Alignment Ablation

Date frozen: 2026-07-25

## Status And Scope

Frozen before any response from this interface. This is a post hoc causal
mechanism test on already open tau-Knowledge trees, not a new held-out policy
result or cross-domain replication. Required-document endpoints are known to
the researchers but never enter a model prompt.

## Question

The existing GPT-5.4 result shows that a scorer viewing complete generated
retrieval trees ranks first roots better than a scorer viewing only first
results. Does that advantage depend on assigning each root its own actual
future branch, or can the scorer obtain it from root semantics and an
arbitrarily attached multiset of futures?

This intervention is distinct from the failed refreshed-belief ablation. That
test moved only generated belief text while leaving every followup query and
retrieved document attached to its original root. The present test moves the
complete future subtree: refreshed beliefs, four followup queries, and all
followup results.

## Frozen Inputs

- V3 smoke artifact SHA-256:
  `61c466ead49a5545abd54dd28e1a2fd7ad070287f5108684a3b2643befc40a01`.
- V3.1 confirmation artifact SHA-256:
  `f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae`.
- Model: OpenRouter `openai/gpt-5.4`, temperature 0, no reasoning.
- Five roots and four followups per root.
- Existing myopic scores and exact root values are reused from the hash-locked
  source; no tree, query, retrieval, belief, or endpoint is regenerated.

## Fixed Intervention

For each task, seed `24361` creates a within-task derangement of the five root
branches. Root `i` keeps:

- its root query;
- its three first retrieved documents;
- the customer opening and initial information-need support; and
- its root index and prompt position.

It receives from a different root:

- the refreshed information-need hypotheses;
- all four followup queries; and
- every followup retrieved document.

Every permutation must be a derangement. The complete future-subtree multiset
must be preserved per task, and canonical records with all future-subtree fields
removed must hash identically before and after transformation.

## Within-Response Pairing And Blinding

Each request contains the correct and deranged trees as anonymous condition A
and condition B. Seed `24362` assigns the correct tree to A on exactly one of
two smoke tasks and 10 of 20 confirmation tasks. The model sees neither
`aligned` nor `shuffled` labels.

The scorer independently returns five root scores and five best-followup
indices for each condition in one strict compact JSON object. This pairing
controls OpenRouter execution drift and makes the treatment comparison
within-response.

The scorer is target-blind and receives no required-document ID, root value,
BM25 score, prior policy score, correctness label, or control outcome.

## Stages

### Serving smoke

- Source: the exact two open V3 smoke trees.
- Requests: exactly 2 logical/physical/HTTP calls.
- Budget: projected `$0.05`, hard cap `$0.25`.
- Pass: both objects parse; exact request count; zero reasoning; both aligned
  and shuffled root-score vectors vary on both tasks; all intervention and
  balanced-blinding invariants pass.

Smoke efficacy is descriptive only. Any serving failure closes this exact
interface without parser repair, reissue, task replacement, or prompt change.

### Causal confirmation

Only a passed smoke authorizes the exact 20 open V3.1 trees:

- exactly 20 logical/physical/HTTP calls;
- projected `$0.40`, hard cap `$1.00`;
- no retries, repairs, replacement responses, or tree regeneration.

## Frozen Causal Criteria

All must pass:

- every serving and intervention invariant;
- aligned aggregate root pairwise accuracy at least `.60`;
- aligned accuracy minus the existing myopic accuracy at least `.05`;
- aligned accuracy minus shuffled accuracy at least `.05`;
- aligned selected-root oracle-tail total at least three required documents
  above shuffled;
- aligned beats shuffled on at least three task endpoints and loses on at most
  two; and
- aligned and shuffled score vectors differ on at least 15/20 tasks.

Aggregate pairwise accuracy uses all endpoint-distinct root pairs. Root policy
value grants the selected root its exact best available continuation, preserving
the existing first-link estimand and removing continuation-choice noise.
Original-order tie breaking is fixed.

Exact task-level one-sided sign-flip values are reported for pairwise-accuracy
and endpoint differences but are descriptive because these trees and endpoints
are already open.

Passing supports the narrow causal statement that correct semantic
root-to-future assignment is load-bearing for the GPT-5.4 ranking gain.
Failure means complete-tree visibility helps, but this experiment does not
isolate correct future assignment from root priors, prompt effects, or generic
future-document scoring.

## Budget And Scheduling

- Maximum newly authorized spend: `$1.25`, with confirmation conditional.
- Live balance before preregistration: `$45.590272884`.
- Protected reserve through Monday: `$25`.
- Headroom above reserve: `$20.590272884`.
- OatML is paused and no cluster job is used.
