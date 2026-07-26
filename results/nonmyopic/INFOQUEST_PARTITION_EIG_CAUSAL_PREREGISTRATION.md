# InfoQuest Semantic-Partition EIG Causal Preregistration

Frozen after the direct-choice shared-action result and before any response or
endpoint on this distinct exact-scoring route.

## Motivation and Question

The direct-choice treatment regenerated all 30 supports but lost to fixed
support (`0/26/4`, mean `-0.1667`). Post hoc inspection of the four disclosed
loss cells found that regenerated supports usually specialized coherently to the
observed answer, while the model selected an intuitively useful next question
without an explicit information objective.

This route removes LLM action choice. It asks whether exact EIG over the LLM's
own weighted semantic hypotheses and answer likelihood partitions makes
path-dependent support regeneration useful.

## Frozen Data and Roles

- disclosed IDs `{0, 1, 4}`, two official settings each;
- six exact fixtures under private SHA-256
  `63248345976e884b9fac127033415fe68bacb3a0e32a32e498285851719251cc`;
- GPT-5.4 non-reasoning owns semantic support, posterior weights, and response
  partitions;
- a deterministic exact scorer owns EIG and action selection;
- Gemini-2.5-Flash non-reasoning answers selected questions under released
  hidden-user prompts;
- GPT-5.4 Mini non-reasoning judges the official five-item checklist;
- no opportunity, development, or holdout record.

## Semantic Likelihood and Exact EIG

For every first root/world/answer, the other four initial roots form the same
candidate bank `A/B/C/D`.

Both compute-matched branches emit exactly:

- `h1..h8`: eight semantic hypotheses;
- `w1..w8`: integer posterior weights in `[1,100]`;
- `a1..a8` through `d1..d8`: answer-cluster labels in `[0,3]`.

Within one action, hypotheses receive the same label iff their predicted
answers are semantically indistinguishable. Labels are local to each action.
The dynamic branch may regenerate K8; the fixed branch must copy the initial K8
exactly. Neither branch chooses an action or sees the official checklist.

For action `x`, the exact scorer normalizes weights, sums probability by answer
cluster, and computes

```text
EIG(x) = -sum_c p(c | x) log p(c | x).
```

This is mutual information under the LLM's deterministic semantic response
partition. The scorer selects maximum EIG with frozen `A,B,C,D` tie order.

The same simulator then answers dynamic- and fixed-EIG actions from matched
histories; one independent call scores immediate, dynamic-two-turn, and
fixed-two-turn official checklist discovery.

## Serving and Mechanics

Serving uses exactly seven synthetic calls: initial, root answer, dynamic
partition, fixed partition, two follow-up answers, and checklist judgment. All
parsers must pass with exact seven physical requests/HTTP attempts, zero
retry/reasoning/forced exits, and cost at most `$0.12`.

Only a serving pass authorizes one exact 159-call mechanics run:

- 3 initial support calls;
- 30 root-answer calls;
- 30 dynamic semantic-partition calls;
- 30 fixed semantic-partition calls;
- 60 paired follow-up-answer calls;
- 6 checklist calls.

The mechanics cap is `$0.95`. Every batch checkpoints before parsing. No
response is repaired, reissued, coerced, deduplicated, imputed, or dropped.

## Conjunctive Scientific Gates

All accounting gates and all of the following must pass:

1. three distinct initial supports;
2. at least 24/30 dynamic supports change;
3. mean dynamic novel-support fraction at least `.50`;
4. at least 24/30 dynamic cells have at least two positive-EIG actions;
5. at least 24/30 fixed cells have at least two positive-EIG actions;
6. dynamic and fixed exact-EIG actions differ in at least 8/30 cells;
7. at least four fixtures use at least two dynamic action labels;
8. mean dynamic next-turn checklist gain at least `.50`;
9. mean dynamic-minus-fixed checklist at least `.15`;
10. dynamic has more paired wins than losses;
11. at least four fixtures have positive mean dynamic-minus-fixed gain;
12. at least four fixtures have dynamic endpoint range at least one.

A pass is the desired first causal link: an LLM-native, path-dependent semantic
belief and likelihood model improves the next BED action while an exact scorer
only performs finite arithmetic. It authorizes only a separately preregistered
ex-ante root-ranking gate. A failure closes this exact route without prompt,
parser, model, threshold, cluster count, action-bank, or disclosed-case repair.

The pre-Monday operational allowance is `$3.07024920`; this route can consume
at most `$1.07`. OatML jobs: `0`.
