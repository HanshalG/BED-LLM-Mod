# UCI Mushroom Projected-Utility GPT-5.4 Mini Confirmation Preregistration

Status: frozen after the exact qualification, failed names-only proposal gate,
passed fresh projected-utility proposal gate, and deterministic full replay, but
before any GPT-5.4 Mini response on the confirmation truths.

## Authorized method

- Fresh seed `24175`; 50 Mushroom catalog rows sampled without replacement.
- Eight paired actions per row.
- Arms: projected-utility GPT depth two, matched-random continuations on the same
  root construction, exact depth one, and exhaustive exact depth two.
- GPT and matched random use two-action branch policies for the first seven rounds;
  all arms use exact depth one on the final action. Exactly 350 logical GPT cells are
  required.
- Up to four roots are fixed by the machine. If fewer than four legal actions remain,
  every remaining legal root is used. If the random-control state cannot form any
  complete two-action query policy, that random arm uses exact depth one for that
  late action. Both exhaustion rules were implemented and dry-tested before
  registration. They do not alter the four-root proposal gate.
- Model `openai/gpt-5.4-mini` through OpenRouter, non-thinking, temperature zero,
  128-token output cap, one bounded correction response.
- Valid branch indexes are retained. Only invalid or missing branches after the
  correction response are projected to the exact minimum-entropy legal continuation.

## Frozen endpoints and gates

Primary endpoint: mean post-action target-entropy AUC. Corroborating endpoint:
truth-class log-posterior AUC. Every comparison is paired by hidden catalog row and
uses 10,000 paired bootstrap replicates.

All producer and independent-audit requirements must pass:

1. Positive paired 95% lower bounds for entropy-AUC and truth-log-AUC gains over
   exact depth one.
2. Positive paired 95% lower bounds for entropy-AUC and truth-log-AUC gains over
   matched random.
3. At least 60% recovery of exhaustive depth two's mean entropy-AUC gain over depth
   one.
4. Specimen collection selected first on at least 75% of GPT trajectories.
5. At most 5% projected logical cells and at most 1% projected branch choices.
6. Fifty distinct paired truths, all four arms complete and legal, exactly 350
   logical GPT cells, zero reasoning tokens, forced exits, and scoring-time LLM
   calls.
7. An independent implementation replays every state, observation, posterior,
   GPT/random policy, exact control, aggregate, projection, and fresh-bootstrap gate
   without LLM calls.

Final class accuracy is descriptive only. No alternate seed, prompt modification,
threshold change, or replacement confirmation follows a failure. Expected cost is
below `$3`, within the existing `$6` run cap. Registered project spend is
`$37.48766596 / $110`, leaving `$72.51233404`.
