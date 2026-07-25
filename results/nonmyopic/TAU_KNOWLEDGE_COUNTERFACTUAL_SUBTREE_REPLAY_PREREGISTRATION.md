# tau-Knowledge Counterfactual Future-Subtree Replay Preregistration

## Status

Frozen before computing any counterfactual endpoint or summary. This is a
zero-call, post hoc mechanism analysis on already-open V3.1 confirmation and
paired future-alignment artifacts. It cannot become a new held-out result.

## Question

The paired future-alignment ablation moved each complete future subtree to a
different first-search root, preserving:

- the root query and first retrieved documents;
- the multiset of refreshed information-need hypotheses;
- the multiset of follow-up queries and results; and
- paired, blinded scoring within one GPT-5.4 response.

Its original analysis evaluated both aligned and shuffled score vectors against
the aligned tree's root endpoints. That tests whether correct alignment helps
the deployed task, but it does not ask whether the shuffled scorer follows the
counterfactual future consequences it was actually shown.

## Exact Counterfactual Endpoint

For every target root, use its unchanged first-result documents plus the
follow-up result documents moved to it by the frozen derangement. Required
document IDs remain the exact hidden endpoint. For each of four displayed
follow-ups, utility is distinct required-document coverage in the union of
target-root first results and moved follow-up results.

No model score, query, document, permutation, tie break, or required-document
label changes. The replay must reproduce both source artifact hashes and every
stored permutation before computing metrics.

## Conditions

Report four score/endpoint pairings:

1. aligned scores on aligned endpoints;
2. shuffled scores on aligned endpoints, reproducing the published mismatch;
3. shuffled scores on counterfactual shuffled endpoints; and
4. aligned scores on counterfactual shuffled endpoints as the symmetric
   mismatched control.

For each pairing report:

- root pairwise ranking accuracy and comparable-pair count;
- selected-root oracle-tail coverage;
- selected root plus the scorer's declared best-follow-up coverage;
- best-follow-up optimal count across all 100 roots; and
- total best-follow-up regret.

Also report task-level exact one-sided sign-flip tests for:

- shuffled own-world minus shuffled aligned-world root accuracy; and
- shuffled own-world minus shuffled aligned-world normalized selected-pair
  utility.

Normalization divides selected-pair coverage by the task's counterfactual
pair-oracle value, using zero when that oracle is zero.

## Frozen Interpretation

Classify the replay as:

- **strong intervention consistency** only if shuffled-on-counterfactual root
  accuracy is at least `.60`, exceeds shuffled-on-aligned accuracy by at least
  `.05`, the task-level one-sided root test is at most `.05`, and at least
  `70/100` declared shuffled follow-ups are counterfactually optimal;
- **directional intervention consistency** if shuffled-on-counterfactual root
  accuracy is at least `.55` and exceeds shuffled-on-aligned accuracy, but any
  strong condition fails; or
- **null/adverse** otherwise.

Regardless of category, this analysis can show that the semantic scorer tracks
displayed future consequences. It cannot isolate refreshed belief text from
the accompanying generated queries and documents, because the intervention
moves the complete subtree.

No OpenRouter call, OatML resource, alternate permutation, subset selection, or
threshold revision follows this replay.
