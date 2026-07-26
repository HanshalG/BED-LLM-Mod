# InfoQuest Cached-Partition EIG V2 Preregistration

Frozen after the V1 mechanics transport failure and before any V2 response or
endpoint.

## Status and Scope

This is a separately versioned disclosed-world development gate. It reuses the
same pre-existing common histories bound by:

- public mechanics SHA-256
  `140f77447da9bd3d83e8a7be7fd45dc8fc792422d4e1cef398f6d8460d13ccc3`;
- private raw SHA-256
  `c4dec013386083afb4279754f4ffb1b0a2f6559bb2a2cf1863d80a849fe6e93e`;
- private fixture SHA-256
  `63248345976e884b9fac127033415fe68bacb3a0e32a32e498285851719251cc`.

As in V1, the cache contributes only three initial K8/root banks and 30
first-root answers. Prior choices, second answers, checklist judgments,
metrics, and V1 partition responses are not measurement inputs.

## Frozen V2 Change

V1 unnecessarily restricted nominal answer-cluster IDs to `0..3`. Five of its
30 checkpointed dynamic responses instead used labels through `7`, treating
each of the eight semantic hypotheses as potentially distinguishable. Since an
eight-particle support can induce at most eight deterministic answer classes,
V2 explicitly permits labels `0..7`.

This is not coercion or modulo mapping. Labels remain local to each candidate
action and exact equality defines the predicted answer partition. Every emitted
label must be an integer in `0..7`; any other value fails closed. V1 remains
closed and reproducible at `0..3`.

Everything else is unchanged: fresh compute-matched GPT-5.4 calls emit dynamic
or exact-copy fixed K8 hypotheses, positive integer weights, and four semantic
partitions; the LLM does not choose an action or see the checklist; exact
deterministic entropy selects `A-D`; Gemini-2.5-Flash answers the paired chosen
continuations; GPT-5.4 Mini judges immediate, dynamic-two-turn, and
fixed-two-turn official checklist discovery.

## Serving, Mechanics, and Gates

The synthetic serving gate remains exactly five physical requests and HTTP
attempts, with all parsers, zero retry/reasoning/forced exits, and cost at most
`$0.12`.

Only a serving pass authorizes one exact 126-call mechanics run capped at
`$0.85`: 30 dynamic partitions, 30 fixed partitions, 60 paired simulator
answers, and 6 checklist judgments. Every batch checkpoints before parsing; no
response is repaired, reparsed, reissued, coerced, imputed, or dropped.

The twelve scientific gates are unchanged from V1:

1. three distinct cached initial supports;
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

A pass remains development evidence only. A positive paper claim requires a
separately preregistered fresh-record confirmation.

The pre-Monday operational allowance is `$2.80390165`; V2 can consume at most
`$0.97`. OatML jobs: `0`.
