# Cleveland Heart Projected-Utility GPT-5.4 Mini Confirmation Preregistration

Status: frozen after the independent exact qualification, failed names-only proposal
gate, passed fresh projected-utility proposal gate, and deterministic end-to-end
replay, but before any GPT-5.4 Mini response on the confirmation truths.

## Authorized method

Fresh S0/S1 evidence established that the utility-grounded policy compiler is
mechanically stable and exactly recovers all 16 strict depth-two workup opportunities
in the balanced proposal gate, with zero projection. This confirmation now tests
realized sequential performance.

- Fresh seed `24167`; 50 Cleveland rows sampled without replacement.
- Eight paired actions per row.
- Arms:
  1. projected-utility GPT depth-two branch policies;
  2. matched-random continuations on the same machine-root construction;
  3. exact depth one; and
  4. exhaustive exact depth two.
- GPT is called for the first seven actions only; all arms use exact depth one on the
  final action. Exactly 350 logical GPT cells are required.
- Up to four roots are machine-fixed. If fewer than four legal actions remain in a
  late unworked state, every remaining legal root is used. This exhaustion behavior
  was implemented and dry-tested before registration; it changes neither the
  four-root proposal gate nor ordinary states.
- GPT-5.4 Mini runs through OpenRouter, non-thinking, temperature zero, 128-token
  output cap, one bounded correction response.
- Valid model branch indexes are retained. Only invalid or missing branches after
  the correction response are projected to the exact minimum-entropy legal
  continuation.

## Frozen endpoints and gates

Primary endpoint: mean post-action target-entropy AUC. Corroborating endpoint:
truth-class log-posterior AUC. Every comparison uses paired truths and 10,000 paired
bootstrap replicates.

All requirements must pass:

1. Positive paired 95% lower bounds for entropy-AUC and truth-log-AUC gains over
   exact depth one.
2. Positive paired 95% lower bounds for entropy-AUC and truth-log-AUC gains over
   matched random.
3. Recovery of at least 60% of exhaustive depth two's mean entropy-AUC gain over
   depth one.
4. Positive paired 95% lower bound for rounds of earlier workup ordering versus
   exact depth one.
5. At most 5% projected logical cells and at most 1% projected branch choices.
6. Fifty distinct paired truths, all four arms complete and legal, exactly 350
   accepted logical GPT cells, zero reasoning tokens, forced exits, and scoring-time
   LLM calls.
7. An independent implementation replays every history, state, observation,
   posterior metric, GPT policy, matched-random policy, exact control, aggregate,
   projection event, and fresh-bootstrap scientific gate without LLM calls.

Final class accuracy is descriptive only. No alternate seed, prompt modification,
threshold change, or replacement confirmation follows a failure. Expected cost is
below `$1.50`, within the existing `$6` run cap. Registered project spend is
`$36.76847341 / $110`, leaving `$73.23152659`.
