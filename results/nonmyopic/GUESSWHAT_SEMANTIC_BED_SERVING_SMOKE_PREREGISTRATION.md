# GuessWhat?! Semantic BED Serving Smoke Preregistration

Date: 2026-07-29

Status: **frozen before any multimodal model response**.

## Authorization

The zero-call source audit passed every frozen gate:

- public manifest SHA-256:
  `027a7f49fc599001eca6fbac0b3fe3e6be021d8cb3512eacad3ee167d0f5fff2`;
- official GuessWhat?! commit:
  `346b7de65d5f18fb8c7d357b7c743d02be429d8a`; and
- official test archive SHA-256:
  `c26c08fbb860786f25ab6940dab135c4f61a6404d0c89bbf5b7a21716306548c`.

This smoke uses only the two frozen serving images. Development and holdout
images remain unopened.

## Visual Interface

Download each source COCO image from the public URL in the manifest and verify
its released dimensions. Create two in-memory overlays:

- candidate overlay: all released candidate bounding boxes labeled `C1...Cn`;
- oracle overlay: only the released hidden target box labeled `TARGET`.

Images and overlays are not committed. Their SHA-256 hashes are reported.
Only source-image and candidate-overlay hashes are public; no target-derived
overlay hash is published.

The planner and likelihood model receive the candidate overlay but no target,
human dialogue, game outcome, or object category annotations. The oracle
receives the target overlay and one generated question but no candidate
categories or human answer. Human dialogue and game outcome fields are used
only to verify the already-frozen source row and are neither supplied to a
model nor scored.

## Exact Ten Calls

Phase one, two concurrent calls:

- model: `openai/gpt-5.4-mini`;
- non-thinking, seed `39100`, temperature `.7`;
- each proposes exactly four distinct visual yes/no questions `Q1...Q4`;
- questions may use visible category, attribute, relation, or position but may
  not refer to candidate IDs, boxes, or labels.

Phase two, two concurrent calls:

- model: `google/gemini-2.5-flash`;
- non-thinking, seed `39200`, temperature `0`;
- for every question and candidate, return an integer estimate in `[0,100]`
  for the probability of a truthful `Yes` answer if that candidate were the
  hidden target.

Phase three, six concurrent calls:

- model: `qwen/qwen3-vl-32b-instruct`;
- non-thinking, seed `39300`, temperature `0`;
- answer `Yes` or `No` for Q1 through Q3 on each target overlay.

Expected requests and HTTP attempts: exactly `10`. No retry, repair,
continuation, reissue, response normalization, model substitution, or
reasoning fallback. Projected cost: `$0.08`; hard run cap: `$0.30`.

## Frozen Gates

All gates are conjunctive:

- exactly 10 accepted requests and 10 HTTP attempts;
- zero retries, provider-error retries, reasoning tokens, and forced exits;
- all ten strict schemas parse;
- both planners return four unique semantic yes/no questions;
- each case has at least two questions whose candidate likelihood range is at
  least 40 points with at least three distinct values;
- each case has at least two questions whose uniform-prior expected `Yes` mass
  lies in `[.2,.8]`;
- the six oracle answers contain both `Yes` and `No`;
- for at least five of six oracle answers, Gemini assigns the released target
  at least 60% `Yes` probability when Qwen answers `Yes`, or at most 40% when
  Qwen answers `No`;
- each case has at least two of three such cross-model-consistent answers;
- total cost is at most `$0.30`; and
- no human dialogue or game outcome is supplied to a model or scored, and no
  policy endpoint, development image, or holdout image is accessed.

Failure closes this exact cross-model interface. There is no prompt, model,
case, seed, or threshold repair. A provider-level rejection of the strict
schema may permit one prospectively frozen transport-only codec amendment,
but semantic failure does not.

Passage authorizes only a separately preregistered first-link ranking-fidelity
experiment on development images. It does not authorize a policy or depth
claim.

## Dry Verification

Before any real response:

- all source and serving tests must pass;
- an exact ten-call deterministic multimodal fixture must pass; and
- implementation, tests, and this preregistration must be committed and pushed.
