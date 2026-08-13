# HiddenBench Adaptive-Elicitation Source Protocol

Date frozen: 2026-08-13

Status: **prospective source admission only; no task values, model responses,
planner scores, or endpoints have been opened under this construction.**

## Question

Does the official HiddenBench release provide a large, immutable population of
semantic hidden-profile worlds with a native information channel that can support
a later test of adaptive, non-myopic elicitation?

This is only a source question. A pass does not establish a depth-two advantage,
calibrated LLM likelihoods, or policy efficacy.

## Binding

- repository: `https://github.com/Yassellee/HiddenBench_ICML`;
- commit: `3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1`;
- tree: `e72388d2d7baf29fdef25807ac00d0bd93dc1ad1`;
- benchmark SHA-256:
  `2815afffca4e470d1dfbc81e625160447df1109ce371968181c9e1e6b90443a3`;
- simulator SHA-256:
  `1728834ea009073a8f7ca14dd4961f387e8f3c7f480a22981f5322c85a87d14f`;
- benchmark loader SHA-256:
  `3961483e5824af418fc3dc4a274a3c9ef61983ee3099cc0e6f619a9147b2daa9`;
- system, first-turn, and later-turn prompt SHA-256 values:
  `38e8a41bb90836b488821729e3072557fa42d3fb69984a50c63bc91aa3c606fe`,
  `526ccec3cecec874ed63498e2eb22bffaa1abd69decedb676371ead9f9df1d8f`,
  and `f83f6be2ba440e4f6fa34b88c92fa2fc84f9e4f6c48e66b4f2e0578bc0756205`;
- MIT license SHA-256:
  `705dcb2b9b5abff9312bd885deae2d3fdcadfe65d4fc0cd0aa3b8b1d8545e354`.

## Frozen Source Gates

All gates are conjunctive.

1. Exact repository commit, tree, and all file hashes match.
2. The benchmark contains exactly 65 tasks with the documented eight-field
   schema and unique IDs and names.
3. Every task has 3--4 unique possible answers, exactly one registered correct
   answer in that list, at least three nonempty shared facts, and 3--4 nonempty
   private facts.
4. Every scenario, task name, fact, answer, and available rationale is nonempty;
   no duplicate fact appears within a task.
5. The released hidden-profile simulator creates one agent per private fact,
   gives a hidden-profile agent all shared facts plus exactly one shuffled private
   fact, and exposes other agents' natural-language messages sequentially.
6. A deterministic hash order using salt `hiddenbench-adaptive-elicitation-v1|`
   creates complete, disjoint splits of 4 mechanics, 12 opportunity, 16
   development, 24 confirmation, and 9 reserve tasks.
7. The public manifest contains only bindings, aggregate counts, and split hashes.
   It serializes no task ID, name, description, fact, answer, rationale, or source
   row.

## Intended Descendant

If this source audit passes, a separate protocol may open only the four mechanics
tasks. Its proposed action is to elicit one role's private evidence. Its latent
hypotheses are the task's answer options, but the response likelihood and
option-conditioned counterfactual completion must be generated semantically by an
LLM; they are not supplied by HiddenBench.

The mechanics gate must establish all of the following before paid development:

- exact response schema and semantic answer-obedience;
- a depth-two changed first action with positive truth-anchored gain on at least
  three of four tasks;
- dynamic branch support or likelihoods that materially differ from an
  answer-free fixed-root control;
- a compute-matched myopic ensemble, random-role control, and common-random-number
  endpoint design;
- outcomes remain sealed until serving and opportunity gates pass.

## Stop Rules

- A source failure closes this exact release and split without repair.
- A source pass authorizes only a separately frozen four-task mechanics protocol.
- It authorizes no OpenRouter call, no development or confirmation task access,
  and no paper efficacy claim.
- HiddenBench's published multi-agent accuracy is background evidence only and is
  not an endpoint for this BED construction.
