# PAPRIKA Clothing Support-Recall Gate

## Purpose

Test the first causal link required by LLM-native non-myopic BED: whether a
target-blind model can identify a question whose realized answer induces a
better next LLM-generated hypothesis support than the question preferred by
one-step EIG. This is a mechanism and scorer gate, not yet a trajectory claim.

The test uses the released PAPRIKA Twenty Questions evaluation split. Clothing
is frozen before model calls because it is the largest evaluation category and
contains dense semantic neighborhoods in which support quality is not reducible
to enumerating the benchmark targets. The generator and ranker never receive
the target or the 30-item evaluation list.

## Frozen protocol

- Data SHA-256:
  `d9b7616d1316886aa1aee84ddc4f126715be4983c50d5160c5c7d223b7e0f16a`.
- Selection seed: `24330`, using `random.Random(seed).shuffle` over the 30
  clothing evaluation indices in file order.
- Serving smoke indices: `219, 220`.
- Activity indices: `217, 221, 28, 20, 27, 21, 214, 26`.
- Untouched confirmation indices:
  `231, 0, 224, 22, 19, 24, 218, 222, 23, 225, 25, 223, 229, 216, 226,
  232, 18, 227, 228, 230`.
- Three fixed target-blind prefix questions are answered by GPT-5.4 Mini from
  the target, before support generation.
- GPT-5.4 non-reasoning generates three independent 12-item open-world supports
  at temperature `.6`; their normalized union is the current support.
- GPT-5.4 non-reasoning proposes three distinct semantic yes/no questions.
- GPT-5.4 Mini labels the current support under each question. The resulting
  support frequency supplies `p(Yes)` and binary immediate EIG.
- For each question and hypothetical answer, GPT-5.4 generates three fresh
  12-item supports at temperature `.6`; normalized unions are the prospective
  branch supports.
- Only after all supports are frozen, GPT-5.4 Mini receives the hidden target
  and records its truthful answer and semantic containment in each support.
- The target-blind GPT-5.4 ranker sees history, current support, predictive
  answer probabilities, and actual branch supports. It scores expected future
  target recall. It does not see target identity, target coverage, or the
  evaluation target list.
- Expected branch-support size is a target-blind non-LLM baseline. Seeded random
  is descriptive only. All model calls have reasoning disabled.

## Stages and stopping rules

The serving smoke is exactly 52 physical requests over two cases. It tests
strict parsing, complete support trees, target-blind ranker serving, exact
request count, and zero reasoning tokens. Its efficacy values cannot change any
prompt, split, threshold, or endpoint.

If smoke passes, the activity gate is exactly 200 physical requests over eight
fresh cases and does not call the ranker. It passes only if:

- at least three current supports omit their targets;
- at least two omitted targets appear in some realized candidate branch; and
- at least three cases have candidate-dependent realized coverage.

Failure closes this clothing route without confirmation.

If activity passes, the confirmation is exactly 520 physical requests over the
20 untouched cases. The primary endpoint is the paired difference in realized
next-support target coverage between the target-blind ranker selection and the
immediate-EIG selection. Frozen confirmation success requires all of:

- exact completion and request count, zero reasoning, and valid supports;
- ranker and immediate EIG select different questions on at least five cases;
- mean paired ranker gain over immediate EIG at least `.10`;
- the seeded 10,000-resample 90% bootstrap lower bound is strictly positive;
- more ranker wins than losses;
- ranker mean coverage is not below expected-support-size selection; and
- candidate-level ranker-score Spearman correlation with realized coverage is
  positive and exceeds both immediate EIG and expected support size.

Bootstrap seed is `24331`. Ties use original candidate order. No target,
question, case, or invalid response may be replaced. Any malformed response or
budget/runtime failure stops the stage without retrying the scientific design.

## Budget

The live account balance was `$66.292031753` when this route was frozen. Reserve
at least `$25` through Monday 2026-07-27 and do not use OatML. The OpenRouter
project ceiling remains `$105.38480269545715`. Per-stage hard caps are `$1`,
`$3`, and `$6`, with projected costs of `$0.50`, `$1.50`, and `$3.50`.
Confirmation is forbidden unless both earlier stages pass.
