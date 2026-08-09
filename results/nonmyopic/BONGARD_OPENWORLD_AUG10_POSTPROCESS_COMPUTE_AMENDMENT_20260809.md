# Bongard OpenWorld August 10 Postprocess Compute Amendment

Frozen: 2026-08-09 (Europe/London), before any August 10 Bongard response,
candidate label, endpoint label, or terminal artifact was opened.

Status: **prospective zero-call execution amendment; authorizes no model request**.

This amendment closes an ordering gap introduced after the original August 10
postprocess was frozen. The postprocessor is the sole mechanics-stage analysis
handoff, but the newer strict compute-matched control audit was not one of its
components. Running that audit separately would create exactly the omission and
double-execution risk the one-shot postprocessor was designed to remove.

It changes no model, effort, prompt, response schema, task, seed, action,
endpoint, policy, threshold, request count, paid wrapper, development
authorization, or budget.

## V2 State Machine

The original disposition rules remain unchanged. A disposition other than an
exact wrapper-bound `mechanics_pass` remains terminal and opens no endpoint or
raw-belief analysis.

For an exact authorized mechanics pass, postprocess V2 runs, in order:

1. the frozen DINOv2 plus SigLIP2 classical-suite adapter;
2. the frozen path-mediation replay;
3. the frozen compute-matched control audit.

The compute audit receives the exact mechanics result and exact August 10 wrapper.
Its own authorizer must succeed before it loads the stage result. It verifies the
stored score decomposition, shuffled continuation permutation, score argmaxes,
and finite endpoint metrics before emitting paired summaries.

Only after all three zero-call analyses checkpoint successfully may the
postprocessor write one terminal `RESULT.json`. That composite binds all four
components: disposition, classical suite, path mediation, and compute-matched
control. The compute component must have:

- interface `bongard-openworld-compute-matched-control-1`;
- status `compute_matched_control_audit_complete`;
- stage `mechanics` and the exact mechanics source path and SHA-256;
- `compute_contract_exact: true`;
- `shuffled_dynamic_depth2` as the strict compute-matched control;
- `history_blind_depth2` as the request-count-matched control;
- `myopic_width` as online-regeneration greedy but not compute matched;
- exactly four task rows and frozen 20,000-draw Brier/log-loss summaries;
- zero model calls and cost, no paid authorization, and no claim-tier authority.

The postprocessor independently reconstructs the complete compute report from the
same stage data and recorded stage authorization before accepting an interrupted
checkpoint. Canonical mismatch fails closed.

## Failure And Exactly-Once Behavior

An exception in any downstream component writes one terminal `FAILURE.json` with
the failed stage and hashes of every earlier completed component. Later invocations
return that failure and do not resume, rerun, or skip ahead. A successful V2 result
means the mechanics compute audit has already run; no separate mechanics audit is
permitted afterward.

As before, the composite record may only restate the paid wrapper's pre-existing
development authorization. It creates no authorization and cannot alter, rescue,
or veto a science gate, claim tier, confirmation decision, classical scope, or
paper headline.

## Bindings

- original August 10 postprocess protocol SHA-256:
  `f7d19ef3a8a48478aa30d4b63541e0c680f74641de70ed3120b25f203d888f5e`;
- unchanged paid August 10 wrapper SHA-256:
  `adf0cede0c14e1ac96206461371f2f53f434f5b748327f9cf93ae0e7f521f9a5`;
- unchanged mechanics disposition SHA-256:
  `870c6fe5ba1a9dd09250193bc36e4bb108708a10dc07bc225dc075d596f4d8bb`;
- unchanged classical-suite adapter SHA-256:
  `8c2bc93b3d416a47c2d1e19112a670f270b79712c22e4ba3d2181308e3d0e032`;
- unchanged path-mediation implementation SHA-256:
  `1aef1c9eb90757bd31fec4beb077ddf79965e1a42b2715b4f7a6788e57e8b912`;
- unchanged compute-matched audit protocol SHA-256:
  `09616a6ed8f8164abf932793fd91d27cf38f86a1691533c75cf9901c08324d9b`;
- unchanged compute-matched audit implementation SHA-256:
  `929eda107f8cb60caf4cd7363f07856e135adaa89f16e946edb710c1a93dfbba`.

This amendment makes zero model calls, costs `$0`, and authorizes no paid call,
rerun, endpoint opening, or new claim.
