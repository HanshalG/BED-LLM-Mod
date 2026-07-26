# InfoQuest Refresh-Continuation Preregistration

Frozen after the oracle-support-score serving line failed and before any
response on this distinct judge-free route.

## Question

Holding the first clarification root, hidden world, and first hidden-user
answer fixed, does allowing GPT-5.4 to regenerate its latent-context support
produce a better next clarification than forcing it to retain the initial
support?

This directly tests non-myopia over the LLM's path-dependent belief dynamics.
The treatment and control differ only in whether the eight-hypothesis support
may change after the first observation.

## Frozen Data and Roles

- disclosed IDs `{0, 1, 4}`, two official settings each;
- six exact fixtures under private SHA-256
  `63248345976e884b9fac127033415fe68bacb3a0e32a32e498285851719251cc`;
- GPT-5.4 non-reasoning for initial support and both continuations;
- Gemini-2.5-Flash non-reasoning under the exact released hidden-user system
  prompts;
- GPT-5.4 Mini non-reasoning for the official five-item checklist endpoint;
- no hidden-truth support judge;
- no opportunity, development, or holdout record.

## Paired Procedure

For each of three ambiguous seeds, GPT-5.4 generates eight concrete latent
contexts and five atomic clarification roots. For every root in both hidden
worlds:

1. the same hidden-user simulator answers the root;
2. **dynamic treatment:** GPT-5.4 may regenerate all eight hypotheses from the
   complete seed/root/answer history and chooses one follow-up;
3. **fixed control:** a separate GPT-5.4 call must copy the original eight
   hypotheses exactly and chooses one follow-up from that fixed support;
4. the same simulator answers both follow-ups from matched histories;
5. one independent checklist call scores immediate, dynamic two-turn, and
   fixed two-turn discovery.

The dynamic and fixed GPT calls are compute-matched: both emit `h1..h8` and one
atomic follow-up under the same output cap. The fixed parser rejects any
changed, reordered, added, or removed hypothesis. Support size is always eight.

All batches are checkpointed before parsing. One optional outer JSON/code fence
is transport-only. No response is repaired, reissued, deduplicated, imputed, or
silently dropped.

## Serving Gate

Run exactly ten synthetic calls:

- two initial-support calls;
- two root-simulator calls;
- two dynamic-refresh calls;
- two compute-matched fixed-support calls;
- one follow-up-simulator call;
- one checklist-judge call.

All parsers must pass with exact ten physical requests/HTTP attempts, zero
retry, reasoning tokens, and forced exits, and cost at most `$0.12`. A failure
stops before disclosed mechanics.

## Mechanics Accounting

The one mechanics run uses:

- 3 initial-support calls;
- 30 root-simulator calls;
- 30 dynamic-refresh calls;
- 30 compute-matched fixed-support calls;
- 60 paired follow-up-simulator calls;
- 6 checklist-judge calls;

for exactly **159 physical requests and HTTP attempts**, capped at `$0.85`.

## Conjunctive Gates

Mechanics/accounting gates must all pass, plus:

1. all three seeds produce distinct initial support hashes;
2. at least 24/30 dynamic branch supports differ from their initial support;
3. mean fraction of new normalized hypotheses is at least `.50`;
4. at least 20/30 dynamic follow-ups differ from fixed follow-ups;
5. mean dynamic next-turn checklist gain is at least `.50`;
6. dynamic continuation beats fixed continuation by at least `.15` checklist
   items over all 30 paired root/world cells;
7. dynamic has more paired wins than losses;
8. at least four of six fixtures have positive mean dynamic-minus-fixed gain;
9. at least four fixtures have dynamic two-turn endpoint range of at least one
   checklist item across roots.

All are conjunctive. A pass establishes the first LLM-native causal link:
path-dependent support regeneration improves the next experimental action
under matched histories and compute. It authorizes only a separately
preregistered ex-ante root-ranking gate. A failure closes this exact route
without prompt, parser, model, threshold, support-size, or disclosed-case
repair.

The previous serving line's `$0.02862995` spend leaves `$3.97137005` of the
pre-Monday allowance. This route can consume at most `$0.97` including serving
and mechanics. OatML jobs: `0`.
