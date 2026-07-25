# BrowseComp-Plus LLM-Native Semantic Mechanics Preregistration

## Purpose

Test the first link directly on five already open mechanics tasks: whether an
LLM-generated non-myopic semantic strategy score ranks realized two-search
evidence better than the same model's direct/root score.

This is not answer-generation benchmarking. The LLM owns the open-world answer
hypotheses, weights, root query strategies, observation-conditioned support
refresh, adaptive second queries, and future value scores. A deterministic
BM25 retriever over each task's official evidence plus hard-negative pool is
the bounded environment transition. Exact official evidence/gold document IDs
are hidden until all model responses are checkpointed.

## Frozen Run

- Tasks: mechanics IDs `286, 1058, 747, 787, 854`.
- Model: non-reasoning `openai/gpt-5.4`, temperature `0`.
- Strict ordered flat lines; no JSON, extraction, repair, or reissue.
- Eight normalized-distinct answer hypotheses with positive integer weights
  summing exactly to `100`.
- Six normalized-distinct root queries per task.
- BM25 top `3`, using at most 30,000 characters per indexed document and
  1,200 characters per returned observation.
- Exactly `55` model calls:
  `5` initial + `30` root refresh + `5` aligned future scorer +
  `5` one-step deranged future scorer + `10` selected-path terminal belief.
- The run stops immediately if the initial five responses fail parsing or
  query diversity. No partial repair.
- Zero model retries, reasoning tokens, or forced exits.
- Cost cap `$0.90`, projected below `$0.55`.
- OpenRouter only; OatML prohibited.

The myopic policy selects the largest direct score. Strategy-d2 selects the
largest direct-plus-aligned-future score. The shuffled scorer rotates each
root's refreshed belief and adaptive query to the next root and is diagnostic,
not part of policy selection.

## Endpoints

For every root:

- immediate value: number of official evidence documents in root top-3;
- future gain: new evidence documents added by adaptive top-3;
- total value: evidence union after both searches;
- gold value: answer-bearing documents in the union.

Report pooled within-task pair accuracy for direct-vs-immediate,
future-vs-future-gain, full-vs-total, direct-vs-total, and
shuffled-full-vs-total. Also report policy-selected exact evidence and final
gold-answer mass.

## Conjunctive Gate

All conditions must pass:

- exact `55` physical and HTTP requests;
- zero retries, reasoning tokens, and forced exits;
- at least `27/30` root beliefs change;
- at least `24/30` adaptive queries differ from roots;
- at least `10/30` branches gain evidence on the second search;
- at least `30/25/30` comparable pairs for
  direct/future/full endpoints;
- direct ranks immediate evidence at least `0.55`;
- aligned future ranks future gain at least `0.55`;
- aligned full ranks total evidence at least `0.58`;
- aligned full exceeds direct-on-total pair accuracy by at least `0.05`;
- strategy-d2 selects a different root on at least `2/5` tasks;
- strategy-d2 beats myopic exact total evidence on at least one task and loses
  on at most one; and
- cost at most `$0.90`.

Failure closes this exact method without rescaling scores, changing BM25,
relaxing parsing, or opening development. A full pass authorizes only a
separately committed small development protocol; holdout remains sealed.
