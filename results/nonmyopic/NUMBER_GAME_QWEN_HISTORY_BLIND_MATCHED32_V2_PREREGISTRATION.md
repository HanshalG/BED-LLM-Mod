# Number Game Qwen History-Blind Matched-32 V2 Preregistration

Date frozen: 2026-07-30

## Purpose

This is the fresh-seed successor to the failed matched history-blind control.
V1 completed all 3,072 strict responses but failed closed before endpoint
scoring because `2/1,536` otherwise valid pools had second-draw novelty `1`
instead of `>=2`.

V2 changes only that auxiliary mechanics rule. Final pooled support size and
each raw draw's validity remain gated. Second-draw novelty is descriptive. The
source cohort, model, prompts, matched update, endpoints, scientific gates,
request count, and budget are unchanged.

V1 responses are permanently banked and may not be scored, reused, repaired,
or combined with V2.

## Frozen Bindings

- V1 public `FAILURE.json` SHA256:
  `f394c889a56fdb20854d28cda0fa2927ae1a8d40e02c54f0f400f3f2757fdcb3`
- V1 controls SHA256:
  `3696476ea4aabd17f92ba06fb5142411369c11cc9aabb91f0c2feeac25437f89`
- source `RESULT.json` SHA256:
  `04177da415498d765baa2b460f6d732a5162b118776df4d5019f7b540bfce36e`
- source `TREES.json` SHA256:
  `8535df7eba5437c564cc6df56816fcee0a6991c3358fdaa6b62c6ca96948da82`
- source `TARGETS.json` SHA256:
  `2fd09b75bcce734e17f237ced9a49e97e13e290e2f5229b9354ad7a8a1bdafd6`
- serving-smoke `RESULT.json` SHA256:
  `0e4ba2e5bedd000d5b22c46e16a91404ec8913ba620ab54db286309fef9b8ef4`

The same first 32 source trees and 33 uniformly weighted canonical targets are
used. The serving smoke is reused because V2 does not change the model, prompt,
schema, parser, or transport path, and V1 itself provided 3,072 additional
strict transport observations.

## Fresh Seeds

V2 control seed schedule:

`9200000 + 1000 * tree_index + 2 * history_index + draw_index`

All 3,072 seeds are new. The branch ordering is unchanged from V1.

## Frozen Mechanics

Required:

- exactly `32` source trees and `1,536` branch slots;
- exactly `3,072` accepted requests;
- HTTP attempts equal accepted requests plus retries;
- at most `32` retries and provider-error retries;
- all `3,072` draws strict JSON;
- every raw draw has at least `16` valid unique extensions;
- every pooled support has at least `24` unique extensions;
- zero reasoning tokens and forced exits;
- run cost at most `$4.25`;
- starting authenticated balance at least `$5.00`.

Second-draw novel contribution is reported as its full distribution, minimum,
mean, and fraction at least two. It is not a mechanics or scientific gate.

Any required mechanics failure stops before canonical scoring. No partial
scoring or response reuse is allowed.

## Frozen Scientific Analysis

Identical to V1:

- exact canonical posterior-predictive MSE;
- exact truth-extension coverage;
- support size;
- first- and second-stage conditional-minus-history-blind intervals;
- changed-root dynamic-minus-fixed prompt-conditioning benefit;
- benefit-to-realized-advantage Spearman diagnostic.

Prompt conditioning is directionally coherent only if:

1. at least `20` source trees have different dynamic and fixed roots;
2. second-stage conditional-minus-history-blind predictive MSE has a 95%
   interval below zero;
3. second-stage conditional-minus-history-blind truth coverage has a 95%
   interval whose lower endpoint is at least zero; and
4. changed-root mean dynamic-minus-fixed prompt benefit has a 95% interval
   above zero.

Bootstrap seed/samples remain `9100000` / `20000`. The source result cannot be
rescued, relabeled, or replaced.

## Budget

- live authenticated balance after V1: `$12.184006309`;
- exact V1 cost projection for V2: approximately `$3.21`;
- V2 cap: `$4.25`;
- expected post-run balance: approximately `$8.97`.

Run V2 once after its wrapper, source/failure/smoke bindings, and synthetic
full-path tests pass and are committed.
